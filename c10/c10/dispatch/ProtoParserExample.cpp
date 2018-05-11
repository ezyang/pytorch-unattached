#include <c10.h>
#include <utility>
#include <c10/dispatch/OpSchemaRegistration.h>
#include <c10/dispatch/OpRegistration.h>
#include <c10/cpu/CPUTensorImpl.h>

namespace c10 {

class NodeProto final {
public:
  template<class T> T attribute(const std::string& attribute_name) const;
};

template<> inline bool NodeProto::attribute<bool>(const std::string& /*attribute_name*/) const {
  return false;
}

template<> inline int NodeProto::attribute<int>(const std::string& /*attribute_name*/) const {
  return 0;
}

template<> inline std::string NodeProto::attribute<std::string>(const std::string& /*attribute_name*/) const {
  return "";
}

template<> inline c10::Tensor NodeProto::attribute<c10::Tensor>(const std::string& /*attribute_name*/) const {
  return c10::tensor<int>({5});
}

namespace details {

template<class OpSchemaDef, class Tuple> struct parse_ final {};
template<class OpSchemaDef, class... Arguments> struct parse_<OpSchemaDef, std::tuple<Arguments...>> final {
  static std::function<void()> call(const std::array<const char*, sizeof...(Arguments)>& parameter_names, const NodeProto& proto) {
    std::tuple<Arguments...> arguments = parse_arguments_(parameter_names, proto, std::index_sequence_for<Arguments...>());
    return [arguments] () {
      // TODO Lookup from dispatch only once, i.e. outside of lambda? Must be optional though, because arguments might change.
      return guts::apply(&Dispatcher::call<OpSchemaDef, std::add_lvalue_reference_t<std::add_const_t<Arguments>>...>, arguments);
    };
  }

private:
  template<size_t... I>
  static std::tuple<Arguments...> parse_arguments_(const std::array<const char*, sizeof...(Arguments)>& parameter_names, const NodeProto& proto, std::index_sequence<I...>) {
    return { proto.attribute<Arguments>(std::get<I>(parameter_names))... };
  }
};
}

template<class OpSchemaDef>
class ProtoParser final {
private:
  using Schema = OpSchema<OpSchemaDef>;

public:
  std::function<void()> parse(const NodeProto& proto) {
    return details::parse_<OpSchemaDef, typename Schema::signature::parameter_types::tuple_type>::call(Schema::signature::parameter_names(), proto);
  }
};

}



using namespace c10;
using c10::cpu::CPU_TENSOR;

namespace op {
struct conditional final {
  using Signature = Tensor(bool, Tensor, Tensor);

  static constexpr std::array<const char*, 3> parameter_names = {
    "condition", "then", "else"
  };
};
}

C10_DEFINE_OP_SCHEMA(op::conditional)

Tensor conditional_kernel(bool conditional, Tensor lhs, Tensor rhs) {
  std::cout << "Called with " << conditional << ", " << lhs.data<int>()[0] << ", " << rhs.data<int>()[0] << std::endl;
  return conditional ? lhs : rhs;
}
C10_REGISTER_OP(op::conditional)
  .kernel(&conditional_kernel)
  .dispatchKey({c10::details::TensorParameterDispatchKey{CPU_TENSOR(), TypeMeta::Id<float>()}, c10::details::TensorParameterDispatchKey{CPU_TENSOR(), TypeMeta::Id<float>()}});



int main() {
  ProtoParser<op::conditional> parser;
  auto op = parser.parse(NodeProto());
  op();
  op();
}
