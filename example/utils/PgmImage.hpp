#ifndef PGM_IMAGE_HPP__
#define PGM_IMAGE_HPP__

#include <ostream>
#include <vector>

class PgmImage {
    public:
        PgmImage(int x, int y, unsigned char max_val = 255);
        PgmImage(int x, int y, std::vector<unsigned char> data, unsigned char max_val = 255);
        ~PgmImage();

        char const * raw() const noexcept;
        size_t raw_size() const noexcept;

        void setData(std::vector<unsigned char> const &data);
        std::vector<unsigned char> &getData() noexcept;
        std::vector<unsigned char> const &getData() const noexcept;

        bool writeToFile(std::string const &name) const;

        friend std::ostream & operator<<(std::ostream &lhs, PgmImage const &rhs);
    private:
        void _writeHeader(std::ostream &os) const;

    private:
        size_t _x;
        size_t _y;
        unsigned int _max;
        std::vector<unsigned char> _data;
};

#endif /* end of include guard: PGM_IMAGE_HPP__ */
