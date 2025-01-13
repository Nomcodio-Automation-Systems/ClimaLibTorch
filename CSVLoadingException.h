#include <exception>
#include <string>

// Custom exception for CSV loading errors
class CSVLoadingException : public std::exception {
private:
	std::string message;

public:
	explicit CSVLoadingException(const std::string& msg) : message("CSVLoadingException: " + msg) {}

	const char* what() const noexcept override {
		return message.c_str();
	}
};
