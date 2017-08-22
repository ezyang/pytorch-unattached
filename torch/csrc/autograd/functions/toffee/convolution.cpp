#include "torch/csrc/autograd/functions/convolution.h"

namespace torch { namespace autograd {

void ConvForward::primspec(PrimSpecContext* ctx, jit::node_list inputs, jit::node_list outputs) {
  toffee::NodeProto* p_n = ctx->graph->add_node();

  // Here we have two cases: Conv and ConvTranspose.
  if (transposed == false) {
    p_n->set_op_type("Conv");
  } else {
    p_n->set_op_type("ConvTranspose");
  }

  p_n->add_input(ctx->node(inputs.at(0)));
  p_n->add_input(ctx->node(inputs.at(1)));
  // TODO: Factor this logic into a helper, and make sure it gets applied
  // consistently. See also batch_normalization.cpp
  if (inputs.at(2)->kind() != jit::kConstant || inputs.at(2)->t(jit::kValue).defined()) {
    p_n->add_input(ctx->node(inputs.at(2)));
  }

  p_n->add_output(ctx->node(outputs.at(0)));
  JIT_ASSERT(outputs.at(1)->type()->kind() == jit::TypeKind::HandleType);

  toffee::AttributeProto* attr;
  // Irritatingly, Caffe2 requires us to specify kernels,
  // but we don't actually have that information directly
  // recorded in ConvForward.  So we have to reverse
  // engineer it from the input types...
  // TODO: dynamic_cast ew
  // attribute kernel(s).
  auto weight_type = inputs.at(1)->type()->cast<jit::TensorType>();
  JIT_ASSERT(weight_type);
  auto weight_size = weight_type->sizes();
  std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());
  attr = p_n->add_attribute();
  if (transposed == false) {
    attr->set_name("kernels");
    for (int kernel : kernel_size) {
      attr->add_ints(kernel);
    }
  } else {
    attr->set_name("kernel");
    JIT_ASSERT(kernel_size.size() >= 1);
    int kernel_val = kernel_size[0];
    for (int k : kernel_size) {
      JIT_ASSERT(k == kernel_val);
    }
    attr->set_i(kernel_val);
  }

  // attribute stride(s).
  attr = p_n->add_attribute();
  if (transposed == false) {
    attr->set_name("strides");
    for (int s : stride) {
      attr->add_ints(s);
    }
  } else {
    attr->set_name("stride");
    JIT_ASSERT(stride.size() >= 1);
    int stride_val = stride[0];
    for (int s : stride) {
      JIT_ASSERT(s == stride_val);
    }
    attr->set_i(stride_val);
  }

  // attribute pad(s).
  attr = p_n->add_attribute();
  if (transposed == false) {
    attr->set_name("pads");
    for (int p : padding) {
      attr->add_ints(p);
    }
    // NB: Caffe2 let's specifying top and bottom pads separately;
    // PyTorch assumes it's symmetric
    for (int p : padding) {
      attr->add_ints(p);
    }
  } else {
    attr->set_name("pad");
    JIT_ASSERT(padding.size() >= 1);
    int padding_val = padding[0];
    for (int p : padding) {
      JIT_ASSERT(p == padding_val);
    }
    attr->set_i(padding_val);
  }

  // attribute dilation(s).
  if (transposed == false) {
    attr = p_n->add_attribute();
    attr->set_name("dilations");
    for (int d : dilation) {
      attr->add_ints(d);
    }
  } else {
    // ConvTranspose in Caffe2 only supports dilation=1.
    for (int d : dilation) {
      JIT_ASSERT(d == 1);
    }
  }

  // Caffe2 does not support output_padding.
  for (int p : output_padding) {
    JIT_ASSERT(p == 0);
  }

  if (transposed == false) {
    attr = p_n->add_attribute();
    attr->set_name("group");
    attr->set_i(groups);
    // ignore benchmark/cudnn_enabled
  }
}

}}
