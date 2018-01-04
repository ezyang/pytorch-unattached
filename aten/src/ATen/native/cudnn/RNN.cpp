#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

namespace at { namespace native {

namespace {
  struct RNNParams {
    cudnnDataType_t datatype;
    cudnnRNNMode_t mode;
    cudnnRNNInputMode_t input_mode;
    int64_t hidden_size;
    int64_t num_layers;
    bool batch_first;
    double dropout;
    bool train;
    cudnnDirectionMode_t bidirectional;
    IntList batch_sizes;
    int64_t dropout_seed;
    Tensor weight_buf;
    Tensor dropout_state;

    // computed
    int64_t num_directions;
    int64_t seq_length;
    int64_t mini_batch;
    int64_t input_size;
    RNNDescriptor rnn_desc;
    std::vector<TensorDescriptor> x_descs;
    std::vector<TensorDescriptor> y_descs;
    TensorDescriptor hx_desc;
    TensorDescriptor hy_desc;
    TensorDescriptor cx_desc;
    TensorDescriptor cy_desc;
    FilterDescriptor w_desc;
  };

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
  */
  std::vector<std::vector<Tensor>>
  get_parameters(const RNNParams& fn, cudnnHandle_t handle, const Tensor& weight_buf) {
    auto cudnn_methods = { cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams };
    std::vector<std::vector<Tensor>> params;
    int64_t num_linear_layers = _num_linear_layers(fn);
    int64_t num_layers = fn.num_directions * fn.num_layers;
    size_t cur_offset = 0;
    for (int64_t layer = 0; layer < num_layers; layer++) {
      std::vector<Tensor> layer_params;
      for (auto cudnn_method : cudnn_methods) {
        for (int64_t linear_id = 0; linear_id < num_linear_layers; linear_id++) {
          FilterDescriptor lin_layer_mat_desc;
          void* matrix_pointer;
          CUDNN_CHECK(cudnn_method(
                handle,
                fn.rnn_desc.desc,
                layer,
                fn.x_descs[0].desc,
                fn.w_desc.desc,
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
          auto offset = offset_bytes / elem_size;

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
            layer_params.emplace_back(std::move(param));
          } else {
            AT_ASSERT(cur_offset == offset, "cur_offset == offset");
          }
          cur_offset = offset + *filter_dim_a[0].data<int>();
        }
      } // for cudnn_method
      params.emplace_back(std::move(layer_params));
    } // for layer
    return params;
  }

  void _copyParams() {
  }
} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r, const Tensor& fn_weight_buf, const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool fn_batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& fn_dropout_state
    ) {

  auto input = input_r;

  RNNParams fn;
  switch (fn_mode) {
    case CUDNN_RNN_RELU:
      fn.mode = CUDNN_RNN_RELU;
      break;
    case CUDNN_RNN_TANH:
      fn.mode = CUDNN_RNN_TANH;
      break;
    case CUDNN_LSTM:
      fn.mode = CUDNN_LSTM;
      break;
    case CUDNN_GRU:
      fn.mode = CUDNN_GRU;
      break;
    default:
      throw std::runtime_error("unrecognized mode"); // TODO
  }
  fn.input_mode = CUDNN_LINEAR_INPUT;
  fn.hidden_size = fn_hidden_size;
  fn.num_layers = fn_num_layers;
  fn.batch_first = fn_batch_first;
  fn.dropout = fn_dropout;
  fn.train = fn_train;
  fn.bidirectional = fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  fn.batch_sizes = fn_batch_sizes;
  fn.dropout_seed = 0;  // doesn't actually affect RNG reset (only set)
  fn.dropout_state = fn_dropout_state;
  fn.weight_buf = fn_weight_buf;
  fn.datatype = getCudnnDataType(input);

  // TODO: hope it's not illegal to have empty batch_sizes count as the
  // optional version

  fn.num_directions = fn.bidirectional ? 2 : 1;

  // TODO: Set device to input
  auto handle = getCudnnHandle();
  auto is_input_packed = fn.batch_sizes.size() != 0;

  if (fn.mode != CUDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  if (fn.batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  // TODO: assert input dim == 3

  if (is_input_packed) {
    fn.seq_length = fn.batch_sizes.size();
    fn.mini_batch = fn.batch_sizes[0];
    fn.input_size = input.size(2);  // -1 originally
  } else {
    fn.seq_length = input.size(0);
    fn.mini_batch = input.size(1);
    fn.input_size = input.size(2);
  }

  // TODO: factor out (_hidden_size)
  std::initializer_list<int64_t> hidden_size = {fn.num_layers * fn.num_directions, fn.mini_batch, fn.hidden_size};
  // TODO: factor out (_output_size)
  std::initializer_list<int64_t> output_size;
  if (fn.batch_sizes.size() != 0) {
    output_size = {input.size(0), fn.hidden_size * fn.num_directions};
  } else {
    output_size = {fn.seq_length, fn.mini_batch, fn.hidden_size * fn.num_directions};
  }

  AT_ASSERT(hx.is_contiguous(), "hx.is_contiguous()");
  AT_ASSERT(!cx.defined() || cx.is_contiguous(), "!cx or cx.is_contiguous()");

  auto x = input.contiguous();
  auto output = input.type().tensor(output_size);
  auto hy = hx.type().tensor(hidden_size);
  Tensor cy;
  if (cx.defined()) {
    cy = cx.type().tensor(hidden_size);
  } else {
    cy = hx.type().tensor(); // Booooh
  }
  auto y = output;

  // init descriptors
  // TODO: need logic to do expensive descriptor set somewhere
  auto dropout_p = fn.train ? fn.dropout : 0;
  DropoutDescriptor dropout_desc;
  dropout_desc.set(handle, dropout_p, fn.dropout_state, fn.dropout_seed);
  //dropout_desc.expensiveSet(handle, fn.dropout, fn.dropout_seed);
  //fn.dropout_state = dropout_desc.state;
  // stop
  fn.rnn_desc.set(handle, fn.hidden_size, fn.num_layers, dropout_desc, fn.input_mode, fn.bidirectional, fn.mode, fn.datatype);

  if (is_input_packed) {
    fn.x_descs = rnn_descriptor_sequence(x, fn.batch_sizes);
    fn.y_descs = rnn_descriptor_sequence(y, fn.batch_sizes);
  } else {
    // TODO: Make sure x[0] has same semantics
    fn.x_descs = rnn_descriptor(x[0], fn.seq_length);
    fn.y_descs = rnn_descriptor(y[0], fn.seq_length);
  }
  fn.hx_desc.set(hx, 5);
  fn.hy_desc.set(hx, 5);
  // TODO: DO NOT use fn.cx_desc/fn.cy_desc if cx is not defined!!
  if (cx.defined()) {
    fn.cx_desc.set(cx, 5);
    fn.cy_desc.set(cx, 5);
  }

  // create the weight buffer and copy the weights into it
  // TODO: eliminate this temporary
  Tensor w;
  /*
  // TODO: implement this as a wrapper on top
  if (!fn.weight_buf.defined()) {
    auto num_weights = get_num_weights(handle, fn.rnn_desc, fn.x_descs[0], fn.datatype);
    // TODO: Double check this; it was transcribed from x.new(num_weights)
    fn.weight_buf = x.type().tensor({num_weights});
    // filters require >= 3 dimensions
    fn.w_desc.set(fn_weight_buf, 3);
    w = fn_weight_buf;
    // this zero might not seem necessary, but it is in the case
    // where biases are disabled; then they won't be copied and must be zero'd.
    // Alternatively, _copyParams could be written more carefully.
    w.zero_();
    auto params = get_parameters(fn, handle, w);
    _copyParams(weight, params);
  } else {
  */
  fn.w_desc.set(fn_weight_buf, 3);
  w = fn_weight_buf;

  if (cx.defined() && !cx.sizes().equals(hidden_size)) {
    std::ostringstream oss;
    oss << "Expected cell size " << IntList{hidden_size} << ", got " << cx.sizes();
    throw std::runtime_error(oss.str());
  }

  size_t workspace_size;
  // TODO: This is annoying, having to put the cudnnTensorDescriptor_t
  // in a contiguous array...
  std::vector<cudnnTensorDescriptor_t> x_descs_arr;
  x_descs_arr.reserve(fn.x_descs.size());
  for (auto& x_desc : fn.x_descs) {
    x_descs_arr.emplace_back(x_desc.desc);
  }
  std::vector<cudnnTensorDescriptor_t> y_descs_arr;
  y_descs_arr.reserve(fn.y_descs.size());
  for (auto& y_desc : fn.y_descs) {
    y_descs_arr.emplace_back(y_desc.desc);
  }
  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle,
        fn.rnn_desc.desc,
        fn.seq_length,
        x_descs_arr.data(),
        &workspace_size
        ));
  // Dropped fn.workspace_size
  // TODO: put this in the correct device???
  Tensor workspace = at::CUDA(kByte).tensor(workspace_size);

  // Swapped requires_grad with train
  Tensor reserve;
  if (fn.train) {
    size_t reserve_size;
    CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
          handle,
          fn.rnn_desc.desc,
          fn.seq_length,
          x_descs_arr.data(),
          &reserve_size
          ));
    reserve = at::CUDA(kByte).tensor(reserve_size);
    // TODO: probably reserve needs to be returned
    CUDNN_CHECK(cudnnRNNForwardTraining(
          handle,
          fn.rnn_desc.desc,
          fn.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          fn.hx_desc.desc, hx.data_ptr(),
          cx.defined() ? fn.cx_desc.desc : nullptr, cx.defined() ? cx.data_ptr() : nullptr,
          fn.w_desc.desc, w.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          fn.hy_desc.desc, hy.data_ptr(),
          cy.defined() ? fn.cy_desc.desc : nullptr, cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0),
          reserve.data_ptr(), reserve.size(0)
          ));
  } else { // inference
    reserve = at::CUDA(kByte).tensor();
    CUDNN_CHECK(cudnnRNNForwardInference(
          handle,
          fn.rnn_desc.desc,
          fn.seq_length,
          x_descs_arr.data(), x.data_ptr(),
          fn.hx_desc.desc, hx.data_ptr(),
          cx.defined() ? fn.cx_desc.desc : nullptr, cx.defined() ? cx.data_ptr() : nullptr,
          fn.w_desc.desc, w.data_ptr(),
          y_descs_arr.data(), y.data_ptr(),
          fn.hy_desc.desc, hy.data_ptr(),
          cy.defined() ? fn.cy_desc.desc : nullptr, cy.defined() ? cy.data_ptr() : nullptr,
          workspace.data_ptr(), workspace.size(0)
          ));

    if (fn.batch_first && !is_input_packed) {
      output.transpose_(0, 1);
    }
  }

  // TODO remove this
  if (!reserve.defined()) {
    throw std::runtime_error("reserve not defined");
  }
  return std::make_tuple(output, hy, cy, reserve);
}

}} // namespace at::native
