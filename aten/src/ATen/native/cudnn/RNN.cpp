#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/MatrixRef.h>

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

namespace at { namespace native {

namespace {
  std::vector<TensorDescriptor> rnn_descriptor_sequence(const Tensor& tensor, IntList batch_sizes) {
    std::vector<TensorDescriptor> descriptors(batch_sizes.size());
    size_t i = 0;
    for (auto batch_size : batch_sizes) {
      // NB: The narrow is solely to adjust the batch size; to do it
      // accurately we would have to adjust the start index as well,
      // but the pointer location isn't actually used so we can skip it.
      // NB: cuDNN RNN API has an undocumented requirement that all
      // tensors have dimension 5.
      descriptors[i].set(tensor.narrow(0, 0, batch_size), 5);
      i++;
    }
    return descriptors;
  }

  std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
    std::vector<TensorDescriptor> descriptors(N);
    for (int64_t i = 0; i < N; i++) {
      descriptors[i].set(tensor, 5);
    }
    return descriptors;
  }


  struct RNNParams {
    // 11 fields to fill in
    cudnnDataType_t datatype;
    cudnnRNNMode_t mode;
    int64_t hidden_size;
    int64_t num_layers;
    bool batch_first;
    double dropout;
    bool train;
    cudnnDirectionMode_t bidirectional;
    IntList batch_sizes;
    Tensor weight_buf;
    Tensor dropout_state;

    // constant (at the moment)
    cudnnRNNInputMode_t input_mode;

    // computed
    int64_t num_directions;
    int64_t seq_length;
    int64_t mini_batch;
    int64_t input_size;

    RNNParams()
      : input_mode(CUDNN_LINEAR_INPUT)
      {}

    // NB: must be called BEFORE you transpose input, because
    // we handle the transposition internally
    void init_computed(TensorGeometry input) {
      auto is_input_packed = batch_sizes.size() != 0;
      if (batch_first && !is_input_packed) {
        input = input.transpose(0, 1);
      }
      num_directions = bidirectional ? 2 : 1;
      if (is_input_packed) {
        seq_length = batch_sizes.size();
        mini_batch = batch_sizes[0];
        input_size = input.size(-1);
      } else {
        seq_length = input.size(0);
        mini_batch = input.size(1);
        input_size = input.size(2);
      }
    }

    void set_mode(int64_t fn_mode) {
      switch (fn_mode) {
        case CUDNN_RNN_RELU:
          mode = CUDNN_RNN_RELU;
          break;
        case CUDNN_RNN_TANH:
          mode = CUDNN_RNN_TANH;
          break;
        case CUDNN_LSTM:
          mode = CUDNN_LSTM;
          break;
        case CUDNN_GRU:
          mode = CUDNN_GRU;
          break;
        default:
          throw std::runtime_error("unrecognized mode"); // TODO
      }
    }

    void set_bidirectional(bool fn_bidirectional) {
      bidirectional = fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    }
  };

  // NB: Doesn't include the weight descriptor
  struct RNNDescriptors {
    DropoutDescriptor dropout_desc;
    RNNDescriptor rnn_desc;
    // NB: this won't actually lay out the tensor descriptor pointers
    // in the right way, so you'll have to preprocess them
    std::vector<TensorDescriptor> x_descs;
    std::vector<TensorDescriptor> y_descs;
    TensorDescriptor hx_desc;
    TensorDescriptor hy_desc;
    TensorDescriptor cx_desc;
    TensorDescriptor cy_desc;

    RNNDescriptors(const RNNParams& fn, cudnnHandle_t handle, Tensor x, Tensor y, Tensor hx, Tensor cx) {
      auto is_input_packed = fn.batch_sizes.size() != 0;
      auto dropout_p = fn.train ? fn.dropout : 0;
      // NB: dropout_seed passed dummy 0, because it isn't actually used
      // when dropout_state is defined.
      dropout_desc.set(handle, dropout_p, fn.dropout_state, 0);
      rnn_desc.set(handle, fn.hidden_size, fn.num_layers, dropout_desc, fn.input_mode, fn.bidirectional, fn.mode, fn.datatype);

      if (is_input_packed) {
        x_descs = rnn_descriptor_sequence(x, fn.batch_sizes);
        y_descs = rnn_descriptor_sequence(y, fn.batch_sizes);
      } else {
        x_descs = rnn_descriptor(x[0], fn.seq_length);
        y_descs = rnn_descriptor(y[0], fn.seq_length);
      }
      hx_desc.set(hx, 5);
      hy_desc.set(hx, 5);
      // TODO: DO NOT use cx_desc/cy_desc if cx is not defined!!
      if (cx.defined()) {
        cx_desc.set(cx, 5);
        cy_desc.set(cx, 5);
      }
    }

    // TODO: This is annoying, having to put the cudnnTensorDescriptor_t
    // in a contiguous array...
    std::vector<cudnnTensorDescriptor_t> get_descs(const std::vector<TensorDescriptor>& descs) {
      std::vector<cudnnTensorDescriptor_t> r;
      r.reserve(descs.size());
      for (auto& desc : descs) {
        r.emplace_back(desc.desc);
      }
      return r;
    }

    std::vector<cudnnTensorDescriptor_t> get_x_descs() {
      return get_descs(x_descs);
    }

    std::vector<cudnnTensorDescriptor_t> get_y_descs() {
      return get_descs(y_descs);
    }
  };

  int64_t get_num_weights(cudnnHandle_t handle, const RNNDescriptor& rnn_desc,
                          const TensorDescriptor& x_desc, cudnnDataType_t datatype) {
    size_t weight_size;
    CUDNN_CHECK(cudnnGetRNNParamsSize(handle, rnn_desc.desc, x_desc.desc, &weight_size, datatype));
    auto elem_size = dataSize(datatype);
    AT_ASSERT(weight_size % elem_size == 0, "cudnnGetRNNParamsSize returned nonsensical weight_size");
    return weight_size / elem_size;
  }

  int64_t _num_linear_layers(const RNNParams& fn) {
    switch(fn.mode) {
      case CUDNN_LSTM:
        return 8;
      case CUDNN_GRU:
        return 6;
      case CUDNN_RNN_RELU:
        return 2;
      case CUDNN_RNN_TANH:
        return 2;
      default:
        at::runtime_error("unknown cuDNN RNN mode %d", mode);
    }
  }

  /*
    TODO: UNTESTED!!!

    Returns weight and bias tensors for each layer of the RNN. These tensors
    are views on the underlying weight buffer allocated by CuDNN.

    Note: for LSTM and GRU, which have multiple parameters of each type (4 and 3, respectively),
          these parameters are concatenated along the first dimension.
          These parameters are returned in a consistent order by CuDNN:
              (reset, forget, cell, output) for LSTM
              (reset, input, new) for GRU
    Args:
        fn: The RNN function object holding the RNN state
        handle: a CuDNN handle
        weight_buf: a 1D tensor containing the CuDNN-allocated weight (or grad_weight) buffer
    Returns:
        parameters: [(weight_ih, weight_hh, bias_ih, bias_hh)*], with length equal to the num_layers.
            This is represented as a pair of vector, and outer-dimension stride
            (NB: Can't return MatrixRef because we need to allocate the underlying tensor)
  */
  std::pair<std::vector<Tensor>, size_t> // stride0
  get_parameters(const RNNParams& fn, const RNNDescriptors& descs, const FilterDescriptor& w_desc, cudnnHandle_t handle, const Tensor& weight_buf) {
    auto cudnn_methods = { cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams };
    std::vector<Tensor> params;
    int64_t num_linear_layers = _num_linear_layers(fn);
    int64_t num_layers = fn.num_directions * fn.num_layers;
    size_t cur_offset = 0;
    size_t global_layer_params_count = 0;
    for (int64_t layer = 0; layer < num_layers; layer++) {
      size_t layer_params_count = 0;
      for (auto cudnn_method : cudnn_methods) {
        for (int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
          FilterDescriptor lin_layer_mat_desc;
          void* matrix_pointer;
          CUDNN_CHECK(cudnn_method(
                handle,
                descs.rnn_desc.desc,
                layer,
                descs.x_descs[0].desc,
                w_desc.desc,
                weight_buf.data_ptr(),
                linear_id,
                lin_layer_mat_desc.desc,
                &matrix_pointer
                ));
          cudnnDataType_t data_type;
          cudnnTensorFormat_t format;
          int nb_dims;
          constexpr int min_dim = 3;
          // TODO: The use of CPU tensor here is a bit goofy in c++
          Tensor filter_dim_a = at::CPU(kInt).tensor(min_dim);
          CUDNN_CHECK(cudnnGetFilterNdDescriptor(
                lin_layer_mat_desc.desc,
                min_dim,
                &data_type,
                &format,
                &nb_dims,
                filter_dim_a.data<int>()
                ));

          AT_ASSERT(nb_dims <= min_dim, "cudnnGetFilterNdDescriptor failed nb_dims (%d) <= min_dim (%d)", nb_dims, min_dim);
          auto elem_size = dataSize(fn.datatype);
          auto offset_bytes = (char*)matrix_pointer - (char*)weight_buf.data_ptr();
          // TODO: make this assert more informative
          AT_ASSERT(offset_bytes % elem_size, "offset_bytes `mod` elem_size");
          size_t offset = offset_bytes / elem_size;

          // for all the RNN types provided by CUDNN, all the ih weights
          // are the same size and are allocated in a contiguous chunk
          // (same for the hh weights, and the ih and hh biases).
          // Since we're storing all the weights in a single tensor anyway,
          // might as well merge the CUDNN ones into a single tensor as well
          if (linear_id == 0 || linear_id == num_linear_layers / 2) {
            AT_ASSERT(*filter_dim_a.prod().data<int>() == *filter_dim_a[0].data<int>(), "filter_dim_a.prod() == filter_dim_a[0]");
            std::initializer_list<int64_t> size = {*filter_dim_a[0].data<int>() * num_linear_layers / 2, *filter_dim_a[2].data<int>()};
            // TODO: Check if this leaks memory
            Tensor param = fn.weight_buf.type().tensor().set_(*fn.weight_buf.storage(), offset, size);
            params.emplace_back(std::move(param));
          } else {
            AT_ASSERT(cur_offset == offset, "cur_offset == offset");
          }
          cur_offset = offset + *filter_dim_a[0].data<int>();
        }
      } // for cudnn_method
      if (layer == 0) {
        global_layer_params_count = layer_params_count;
      } else {
        AT_ASSERT(global_layer_params_count == layer_params_count, "%d (global) != %d", global_layer_params_count, layer_params_count);
      }
    } // for layer
    return std::make_pair(params, global_layer_params_count);
  }

  void _copyParams() {
  }

  std::vector<int64_t> _input_size(const RNNParams& fn, const Tensor& input) {
    if (fn.batch_sizes.size() != 0) {
      return {input.size(0), fn.input_size};
    } else {
      return {fn.seq_length, fn.mini_batch, fn.input_size};
    }
  }

  std::vector<int64_t> _hidden_size(const RNNParams& fn) {
    return {fn.num_layers * fn.num_directions, fn.mini_batch, fn.hidden_size};
  }

  std::vector<int64_t> _output_size(const RNNParams& fn, const Tensor& input) {
    if (fn.batch_sizes.size() != 0) {
      return {input.size(0), fn.hidden_size * fn.num_directions};
    } else {
      return {fn.seq_length, fn.mini_batch, fn.hidden_size * fn.num_directions};
    }
  }

} // anonymous namespace

// We explicitly need to support this use-case, as networks that use
// weight-tying (c.f. #3751) need to be packed every iteration.
std::tuple<Tensor, Tensor, Tensor, Tensor> _cudnn_rnn_unflattened(
    const Tensor& input, TensorList weights_arr, int64_t weights_stride0, const Tensor& hx, const Tensor& cx,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout,
    bool train, bool bidirectional, IntList batch_sizes,
    const Tensor& dropout_state
    ) {
  MatrixRef<Tensor> weights{weights_arr, static_cast<size_t>(weights_stride0)};

  auto weight = Tensor{};
  return at::_cudnn_rnn(input, weight, hx, cx, mode, hidden_size, num_layers,
                        batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

// NB: when fn_batch_sizes is empty, that means no batch sizes was specified
std::tuple<Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r, const Tensor& fn_weight_buf, const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool fn_batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state
    ) {

  auto input = input_r;

  RNNParams fn;
  fn.set_mode(fn_mode);
  fn.hidden_size = fn_hidden_size;
  fn.num_layers = fn_num_layers;
  fn.batch_first = fn_batch_first;
  fn.dropout = fn_dropout;
  fn.train = fn_train;
  fn.set_bidirectional(fn_bidirectional);
  fn.batch_sizes = fn_batch_sizes;
  fn.dropout_state = fn_dropout_state;
  fn.weight_buf = fn_weight_buf;
  fn.datatype = getCudnnDataType(input);
  fn.init_computed(TensorGeometry{input});

  // TODO: Set device to input

  if (fn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  // TODO: can batch_first be a wrapper around this function?
  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (fn_batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn);
  auto output_size = _output_size(fn, input);

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  auto output = input.type().tensor(output_size);
  auto hy = hx.type().tensor(hidden_size);
  Tensor cy;
  if (cx.defined()) {
    cy = cx.type().tensor(hidden_size);
  } else {
    cy = hx.type().tensor(); // NB: Not allowed to return undefined tensors
  }
  auto y = output;

  auto handle = getCudnnHandle();
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(fn.weight_buf, 3);

  if (cx.defined() && !cx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected cell size " << IntList{hidden_size} << ", got " << cx.sizes();
    throw std::runtime_error(oss.str());
  }

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc,
        fn.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  Tensor workspace = at::CUDA(kByte).tensor(workspace_size);

  Tensor reserve;
  // NB: Previously, the test was for fn.requires_grad, but we don't have
  // this information.  Use 'train' as a proxy.
  if (fn.train) {
    size_t reserve_size;
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
          handle,
          descs.rnn_desc.desc,
          fn.seq_length,
          x_descs_arr.data(),
          &reserve_size
          ));
    reserve = at::CUDA(kByte).tensor(reserve_size);
    CUDNN_CHECK(cudnnRNNForwardTraining(
          handle,
          descs.rnn_desc.desc,
          fn.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          descs.hx_desc.desc, hx.data_ptr(),
          cx.defined() ? descs.cx_desc.desc : nullptr, cx.defined() ? cx.data_ptr() : nullptr,
          w_desc.desc, fn_weight_buf.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          descs.hy_desc.desc, hy.data_ptr(),
          cy.defined() ? descs.cy_desc.desc : nullptr, cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0),
          reserve.data_ptr(), reserve.size(0)
          ));
  } else { // inference
    reserve = at::CUDA(kByte).tensor();
    CUDNN_CHECK(cudnnRNNForwardInference(
          handle,
          descs.rnn_desc.desc,
          fn.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          descs.hx_desc.desc, hx.data_ptr(),
          cx.defined() ? descs.cx_desc.desc : nullptr, cx.defined() ? cx.data_ptr() : nullptr,
          w_desc.desc, fn_weight_buf.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          descs.hy_desc.desc, hy.data_ptr(),
          cy.defined() ? descs.cy_desc.desc : nullptr, cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0)
          ));

  }

  if (fn.batch_first && !is_input_packed) {
    output.transpose_(0, 1);
  }

  return std::make_tuple(output, hy, cy, reserve);
}

std::tuple<Tensor, Tensor, Tensor> _cudnn_rnn_backward_grad(
    const Tensor& input_r, const Tensor& fn_weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output_r, const Tensor& grad_output_r, const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool fn_batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state, const Tensor& fn_reserve
    ) {

  auto input = input_r;
  auto grad_output = grad_output_r;
  auto output = output_r;

  RNNParams fn;
  fn.set_mode(fn_mode);
  fn.input_mode = CUDNN_LINEAR_INPUT;
  fn.hidden_size = fn_hidden_size;
  fn.num_layers = fn_num_layers;
  fn.batch_first = fn_batch_first;
  fn.dropout = fn_dropout;
  fn.train = fn_train;
  fn.set_bidirectional(fn_bidirectional);
  fn.batch_sizes = fn_batch_sizes;
  fn.dropout_state = fn_dropout_state;
  fn.weight_buf = fn_weight_buf;
  fn.datatype = getCudnnDataType(input);
  fn.num_directions = fn.bidirectional ? 2 : 1;
  fn.init_computed(TensorGeometry{input});

  // TODO: Set device to input
  auto handle = getCudnnHandle();

  if (fn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (fn_batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn, input);
  auto hidden_size = _hidden_size(fn);
  auto output_size = _output_size(fn, input);

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  auto dy = grad_output.contiguous();
  auto y = output;
  auto w = fn.weight_buf;
  auto dx = input.type().tensor(input.sizes()); // TODO: more compact way of saying this
  auto dhy = grad_hy.contiguous().view(hidden_size);
  auto dcy = grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
  auto dhx = hx.type().tensor(hidden_size);
  auto dcx = cx.defined() ? cx.type().tensor(hidden_size) : hx.type().tensor(); // Boooh

  if (!fn.train) {
    throw std::runtime_error("backward_grad can only be called in training mode");
  }
  if (!input.sizes().equals(input_size)) {
    std::ostringstream oss;
    oss << "Expected input size " << IntList{input_size} << ", got " << input.sizes();
    throw std::runtime_error(oss.str());
  }
  if (!output.sizes().equals(output_size)) {
    std::ostringstream oss;
    oss << "Expected output size " << IntList{output_size} << ", got " << output.sizes();
    throw std::runtime_error(oss.str());
  }
  if (hx.defined() && !hx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected hidden size " << IntList{hidden_size} << ", got " << hx.sizes();
    throw std::runtime_error(oss.str());
  }
  if (cx.defined() && !cx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected cell size " << IntList{hidden_size} << ", got " << cx.sizes();
    throw std::runtime_error(oss.str());
  }
  if (dhy.defined() && !dhy.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected d_hidden size " << IntList{hidden_size} << ", got " << dhy.sizes();
    throw std::runtime_error(oss.str());
  }
  if (dcy.defined() && !dcy.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected d_cell size " << IntList{hidden_size} << ", got " << dcy.sizes();
    throw std::runtime_error(oss.str());
  }
  if (!dhy.is_cuda() || !dy.is_cuda() || (dcy.defined() && !dcy.is_cuda())) {
    throw std::runtime_error("Gradients aren't CUDA tensors");
  }

  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(fn.weight_buf, 3);

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc,
        fn.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  // TODO: put this in the correct device???
  Tensor workspace = at::CUDA(kByte).tensor(workspace_size);

  CUDNN_CHECK(cudnnRNNBackwardData(
        handle,
        descs.rnn_desc.desc,
        fn.seq_length,
        y_descs_arr.data(), y.data_ptr(),
        y_descs_arr.data(), dy.data_ptr(),
        descs.hy_desc.desc, dhy.data_ptr(),
        cx.defined() ? descs.cy_desc.desc : nullptr, cx.defined() ? dcy.data_ptr() : nullptr,
        w_desc.desc, w.data_ptr(),
        descs.hx_desc.desc, hx.data_ptr(),
        cx.defined() ? descs.cx_desc.desc : nullptr, cx.defined() ? cx.data_ptr() : nullptr,
        x_descs_arr.data(), dx.data_ptr(),
        descs.hx_desc.desc, dhx.data_ptr(),
        cx.defined() ? descs.cx_desc.desc : nullptr, cx.defined() ? dcx.data_ptr() : nullptr,
        workspace.data_ptr(), workspace.size(0),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));

  if (fn.batch_first && !is_input_packed) {
    dx = dx.transpose_(0, 1);
  }

  return std::make_tuple(dx, dhx, dcx); // TODO
}

// NB: This MUST BE CALLED AFTER _cudnn_rnn_backward_grad.
// We'll give a user friendly combined function...
Tensor _cudnn_rnn_backward_weight(
    // TODO: I think tensor geometry sufficient for weight_buf
    const Tensor& input_r, const Tensor& fn_weight_buf, const Tensor& hx, const Tensor& cx,
    const Tensor& output_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool fn_batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state, const Tensor& fn_reserve
    ) {

  auto input = input_r;
  auto output = output_r;

  RNNParams fn;
  fn.set_mode(fn_mode);
  fn.hidden_size = fn_hidden_size;
  fn.num_layers = fn_num_layers;
  fn.batch_first = fn_batch_first;
  fn.dropout = fn_dropout;
  fn.train = fn_train;
  fn.set_bidirectional(fn_bidirectional);
  fn.batch_sizes = fn_batch_sizes;
  fn.dropout_state = fn_dropout_state;
  fn.weight_buf = fn_weight_buf;
  fn.datatype = getCudnnDataType(input);
  fn.init_computed(TensorGeometry{input});

  auto handle = getCudnnHandle();

  if (fn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  auto is_input_packed = fn_batch_sizes.size() != 0;
  if (fn_batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn, input);
  auto hidden_size = _hidden_size(fn);

  if (!fn.train) {
    throw std::runtime_error("backward_grad can only be called in training mode");
  }
  if (!input.sizes().equals(input_size)) {
    std::ostringstream oss;
    oss << "Expected input size " << IntList{input_size} << ", got " << input.sizes();
    throw std::runtime_error(oss.str());
  }
  if (hx.defined() && !hx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected hidden size " << IntList{hidden_size} << ", got " << hx.sizes();
    throw std::runtime_error(oss.str());
  }
  // TODO: the above were the only checks in rnn.py, but it doesn't seem
  // like these checks are enough

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  const auto& y = output;
  auto dw = fn_weight_buf.type().tensor(fn_weight_buf.sizes()).zero_();

  RNNDescriptors descs(fn, handle, x, y, hx, cx);

  FilterDescriptor w_desc;
  w_desc.set(fn.weight_buf, 3);

  size_t workspace_size;
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        descs.rnn_desc.desc,
        fn.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  Tensor workspace = at::CUDA(kByte).tensor(workspace_size);

  CUDNN_CHECK(cudnnRNNBackwardWeights(
        handle,
        descs.rnn_desc.desc,
        fn.seq_length,
        x_descs_arr.data(), x.data_ptr(),
        descs.hx_desc.desc, hx.data_ptr(),
        y_descs_arr.data(), y.data_ptr(),
        workspace.data_ptr(), workspace.size(0),
        w_desc.desc, dw.data_ptr(),
        fn_reserve.data_ptr(), fn_reserve.size(0)
        ));
  return dw;
}

// We need this dispatcher because _cudnn_rnn_backward_weight has a stringent
// ordering requirement with _cudnn_rnn_backward_grad
std::tuple<Tensor, Tensor, Tensor, Tensor> _cudnn_rnn_backward(
    const Tensor& input, const Tensor& weight, const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output, const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout,
    bool train, bool bidirectional, IntList batch_sizes,
    const Tensor& dropout_state, const Tensor& reserve,
    std::array<bool, 4> output_mask
    ) {
  Tensor dx, dhx, dcx, dw;
  // NB: unconditionally compute this gradient, because it mutates reserve
  std::tie(dx, dhx, dcx) = at::native::_cudnn_rnn_backward_grad(input, weight, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve);
  if (output_mask[3]) {
    dw = at::native::_cudnn_rnn_backward_weight(input, weight, hx, cx, output, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve);
  }
  return std::tuple<Tensor, Tensor, Tensor, Tensor>{dx, dhx, dcx, dw};
}

}} // namespace at::native
