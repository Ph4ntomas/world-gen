#include <iostream>
#include <iomanip>
#include <limits>

int main(void)
{
    std::cout << std::hexfloat;
    std::cout << std::numeric_limits<double>::min() << std::endl;
    std::cout << std::numeric_limits<double>::max() << std::endl;
    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1) << std::endl;
    std::cout << std::numeric_limits<double>::min() << std::endl;
    std::cout << std::numeric_limits<double>::max() << std::endl;
    return 0;
}
