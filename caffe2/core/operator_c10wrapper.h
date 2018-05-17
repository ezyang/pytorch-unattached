#pragma once

#include "operator.h"
#include <c10/dispatch/Dispatcher.h>

namespace caffe2 {

template<class OpSchemaDef, class Context>
class C10OperatorWrapper final : public Operator<Context> {
  using Schema = c10::OpSchema<OpSchemaDef>;
public:
 C10OperatorWrapper(const OperatorDef& operator_def, Workspace* ws)
     : Operator<Context>(operator_def, ws) {}

 USE_OPERATOR_CONTEXT_FUNCTIONS;

 bool RunOnDevice() override {
   RunOnDevice_(std::make_index_sequence<Schema::signature::num_args>());
   return true;
 }

private:
 template<size_t... InputIndex>
 void RunOnDevice_(std::index_sequence<InputIndex...>) {
   auto output = c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)...);
   // TODO Return output, but avoid pre-allocating output
   Output(0)->swap(output);
 }
};

template<class OpSchemaDef, class Context>
class C10OperatorWrapper2 final : public Operator<Context> {
    using Schema = c10::OpSchema<OpSchemaDef>;
public:
    C10OperatorWrapper2(const OperatorDef& operator_def, Workspace* ws)
            : Operator<Context>(operator_def, ws) {}

    USE_OPERATOR_CONTEXT_FUNCTIONS;

    bool RunOnDevice() override {
        RunOnDevice_(std::make_index_sequence<Schema::signature::num_args - 1>());
        return true;
    }

private:
    template<size_t... InputIndex>
    void RunOnDevice_(std::index_sequence<InputIndex...>) {
        /*auto output =*/ c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)..., Output(0));
        // TODO Return output, but avoid pre-allocating output
        //Output(0)->swap(output);
    }
};

CAFFE_DECLARE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(OpSchemaDef, Name)           \
  CAFFE_REGISTER_CLASS(C10OperatorRegistry, Name, C10OperatorWrapper<OpSchemaDef, CPUContext>)

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_2(OpSchemaDef, Name)           \
  CAFFE_REGISTER_CLASS(C10OperatorRegistry, Name, C10OperatorWrapper2<OpSchemaDef, CPUContext>)

}
