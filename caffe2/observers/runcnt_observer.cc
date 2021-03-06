#include "runcnt_observer.h"

namespace caffe2 {

RunCountOperatorObserver::RunCountOperatorObserver(
    IOperatorBase* op,
    RunCountNetObserver* netObserver)
    : ObserverBase<IOperatorBase>(op), netObserver_(netObserver) {
  CAFFE_ENFORCE(netObserver_, "Observers can't operate outside of the net");
}

std::string RunCountNetObserver::debugInfo() {
#if CAFFE2_ANDROID
  // workaround
  int foo = cnt_;
  return "This operator runs " + caffe2::to_string(foo) + " times.";
#else
  return "This operator runs " + caffe2::to_string(cnt_) + " times.";
#endif
}

void RunCountNetObserver::Start() {}

void RunCountNetObserver::Stop() {}

void RunCountOperatorObserver::Start() {
  ++netObserver_->cnt_;
}
void RunCountOperatorObserver::Stop() {}

std::unique_ptr<ObserverBase<IOperatorBase>> RunCountOperatorObserver::rnnCopy(
    IOperatorBase* subject,
    int rnn_order) const {
  return std::unique_ptr<ObserverBase<IOperatorBase>>(
      new RunCountOperatorObserver(subject, netObserver_));
}

} // namespace caffe2
