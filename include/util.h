#ifndef DISTLR_UTIL_H_
#define DISTLR_UTIL_H_

#include <string>
#include <vector>

namespace distlr {

std::vector<std::string> Split(std::string line, char sparator);

int ToInt(const char* str);

int ToInt(const std::string& str);

float ToFloat(const char* str);

float ToFloat(const std::string& str);

} // namespace distlr

#endif  // DISTLR_UTIL_H_
