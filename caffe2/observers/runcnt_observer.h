#pragma once

#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"
#include "caffe2/observers/operator_attaching_net_observer.h"

namespace caffe2 {

class RunCountNetObserver;
class RunCountOperatorObserver final : public ObserverBase<IOperatorBase> {
 public:
  explicit RunCountOperatorObserver(IOperatorBase* op) = delete;
  RunCountOperatorObserver(IOperatorBase* op, RunCountNetObserver* netObserver);
  ~RunCountOperatorObserver() {}
  std::unique_ptr<ObserverBase<IOperatorBase>> rnnCopy(
      IOperatorBase* subject,
      int rnn_order) const override;

 private:
  void Start() override;
  void Stop() override;

 private:
  RunCountNetObserver* netObserver_;
};

class RunCountNetObserver final : public OperatorAttachingNetObserver<
                                      RunCountOperatorObserver,
                                      RunCountNetObserver> {
 public:
  explicit RunCountNetObserver(NetBase* subject_)
      : OperatorAttachingNetObserver<
            RunCountOperatorObserver,
            RunCountNetObserver>(subject_, this),
        cnt_(0) {}
  ~RunCountNetObserver() {}

  std::string debugInfo() override;

  friend class RunCountOperatorObserver;

 private:
  void Start() override;
  void Stop() override;

 protected:
  std::atomic<int> cnt_;
};

} // namespace caffe2
