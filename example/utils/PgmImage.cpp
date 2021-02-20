#include <iostream>
#include <fstream>

#include "./PgmImage.hpp"

PgmImage::PgmImage(int x, int y, unsigned char max_val):
    _x(x), _y(y), _max(max_val),
    _data(x * y, 0)
{
    _data.shrink_to_fit();
}

PgmImage::PgmImage(int x, int y, std::vector<unsigned char> data, unsigned char max_val):
    _x(x), _y(y), _max(max_val),
    _data(data)
{
    _data.resize(x * y, 0);
    _data.shrink_to_fit();
}

PgmImage::~PgmImage() {}

char const * PgmImage::raw() const noexcept {
    return reinterpret_cast<char const *>(_data.data());
}
size_t PgmImage::raw_size() const noexcept {
    return _data.size();
};

void PgmImage::setData(std::vector<unsigned char> const &data) {
    _data = data;

    _data.resize(_x * _y, 0);
    _data.shrink_to_fit();
}

std::vector<unsigned char> &PgmImage::getData() noexcept { return _data; }
std::vector<unsigned char> const &PgmImage::getData() const noexcept { return _data; }

bool PgmImage::writeToFile(std::string const &name) const {
    std::ofstream fstr(name);

    fstr << *this;

    return true;
}

void PgmImage::_writeHeader(std::ostream &os) const {
    os << "P5" << std::endl;
    os << _x << " " << _y << std::endl;
    os << _max << std::endl;
}

std::ostream & operator<<(std::ostream &lhs, PgmImage const &rhs) {
    rhs._writeHeader(lhs);

    lhs.write(rhs.raw(), rhs.raw_size());
    return lhs;
}

