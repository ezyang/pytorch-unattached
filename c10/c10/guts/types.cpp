#include "caffe2/core/typeid.h"
#include <atomic>


C10_KNOWN_TYPE(float) ;
C10_KNOWN_TYPE(int) ;
C10_KNOWN_TYPE(std::string) ;
C10_KNOWN_TYPE(bool) ;
C10_KNOWN_TYPE(uint8_t) ;
C10_KNOWN_TYPE(int8_t) ;
C10_KNOWN_TYPE(uint16_t) ;
C10_KNOWN_TYPE(int16_t) ;
C10_KNOWN_TYPE(int64_t) ;
C10_KNOWN_TYPE(double) ;
C10_KNOWN_TYPE(char) ;
C10_KNOWN_TYPE(std::unique_ptr<std::mutex>) ;
C10_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>) ;
C10_KNOWN_TYPE(std::vector<int32_t>) ;
C10_KNOWN_TYPE(std::vector<int64_t>) ;
C10_KNOWN_TYPE(std::vector<unsigned long>) ;
C10_KNOWN_TYPE(bool*) ;
C10_KNOWN_TYPE(char*) ;
C10_KNOWN_TYPE(int*) ;

