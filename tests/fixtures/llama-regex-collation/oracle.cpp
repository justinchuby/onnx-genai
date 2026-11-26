// Reproduce the C-locale std::regex<char> bracket-expression oracle.
// Build: clang++ -std=c++17 oracle.cpp -o /tmp/llama-collation-oracle
// Input: one hex-encoded pattern per line. Output: pattern, valid/invalid,
// and the hex bytes whose one-byte strings match. No runtime C++ dependency.
#include <iomanip>
#include <iostream>
#include <locale>
#include <regex>
#include <sstream>
#include <string>

static std::string unhex(const std::string &hex) {
    std::string value;
    for (size_t i = 0; i < hex.size(); i += 2) {
        value.push_back(static_cast<char>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return value;
}
int main() {
    std::locale::global(std::locale::classic());
    std::string hex;
    while (std::getline(std::cin, hex)) {
        const auto pattern = unhex(hex);
        std::cout << hex << '\t';
        try {
            std::regex regex;
            regex.imbue(std::locale::classic());
            regex.assign(pattern, std::regex::ECMAScript);
            std::cout << "valid\t";
            bool any = false;
            for (unsigned byte = 0; byte < 256; ++byte) {
                const std::string input(1, static_cast<char>(byte));
                const bool full = !pattern.empty() && pattern.front() == '^' && pattern.back() == '$';
                if (full ? std::regex_match(input, regex) : std::regex_search(input, regex)) {
                    any = true;
                    std::cout << std::hex << std::setfill('0') << std::setw(2) << byte;
                }
            }
            if (!any) std::cout << '-';
        } catch (const std::regex_error &) {
            std::cout << "invalid\t-";
        }
        std::cout << '\n';
    }
}
