//
//  tensor_index.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_index_h
#define guochu_gtensor_tensor_index_h

#include "tensor_index_operation.hpp"
#include "gtensor_utility.h"
#include "tensor_label.h"
#include <array>
#include <boost/mpl/int.hpp>
#include <boost/static_assert.hpp>
#include <boost/serialization/base_object.hpp>

namespace guochu {
    namespace tensor {
        
        template <size_t N>
        class tensor_index
        :public std::array<size_t, N>{
            typedef tensor_index<N> self_type;
            typedef std::array<size_t, N> super_type;
        public:
            typedef size_t size_type;
            typedef typename super_type::iterator iterator;
            typedef typename super_type::const_iterator const_iterator;
            tensor_index() : super_type(){}
            tensor_index(const super_type& sup) : super_type(sup) {}
            
            const self_type& operator=(const super_type& sup)
            {
                super_type::operator=(sup);
                return *this;
            }
            
            explicit tensor_index(size_t i);
            explicit tensor_index(size_t i, size_t j);
            explicit tensor_index(size_t i, size_t j, size_t k);
            explicit tensor_index(size_t i, size_t j, size_t k, size_t l);
            explicit tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m);
            explicit tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n);
            explicit tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t x);
            explicit tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t x, size_t y);
            
            const self_type& operator+=(const self_type &A);
            const self_type& operator-=(const self_type &A);
            void write(size_t precision = 6)const;
            
            template<typename Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
                ar & boost::serialization::base_object<super_type>(*this);
            }
            
        };
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i)
        {
            BOOST_STATIC_ASSERT(N == 1);
            this->operator[](0) = i;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j)
        {
            BOOST_STATIC_ASSERT(N == 2);
            this->operator[](0) = i, this->operator[](1) = j;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j, size_t k)
        {
            BOOST_STATIC_ASSERT(N == 3);
            this->operator[](0) = i, this->operator[](1) = j,
            this->operator[](2) = k;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j, size_t k, size_t l)
        {
            BOOST_STATIC_ASSERT(N == 4);
            this->operator[](0) = i, this->operator[](1) = j,
            this->operator[](2) = k, this->operator[](3) = l;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m)
        {
            BOOST_STATIC_ASSERT(N == 5);
            this->operator[](0) = i, this->operator[](1) = j,
            this->operator[](2) = k, this->operator[](3) = l;
            this->operator[](4) = m;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n)
        {
            BOOST_STATIC_ASSERT(N == 6);
            this->operator[](0) = i, this->operator[](1) = j,
            this->operator[](2) = k, this->operator[](3) = l;
            this->operator[](4) = m, this->operator[](5) = n;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t x)
        {
            BOOST_STATIC_ASSERT(N == 7);
            this->operator[](0) = i, this->operator[](1) = j,
            this->operator[](2) = k, this->operator[](3) = l;
            this->operator[](4) = m, this->operator[](5) = n;
            this->operator[](6) = x;
        }
        
        template <size_t N>
        tensor_index<N>::tensor_index(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t x, size_t y)
        {
            BOOST_STATIC_ASSERT(N == 8);
            this->operator[](0) = i, this->operator[](1) = j,
            this->operator[](2) = k, this->operator[](3) = l;
            this->operator[](4) = m, this->operator[](5) = n;
            this->operator[](6) = x, this->operator[](7) = y;
        }
        
        
        template <size_t N>
        const typename tensor_index<N>::self_type&
        tensor_index<N>::operator+=(const self_type& A)
        {
            for (size_t i = 0; i < N; ++i) {
                this->operator[](i) += A[i];
            }
            return *this;
        }
        
        template <size_t N>
        const typename tensor_index<N>::self_type&
        tensor_index<N>::operator-=(const self_type& A)
        {
            for (size_t i = 0; i < N; ++i) {
                assert(this->operator[](i) > A[i]);
                this->operator[](i) -= A[i];
            }
            return *this;
        }
        
        template <size_t N>
        void tensor_index<N>::write(size_t precision)const
        {
            size_t dim = N;
            return detail::print_mult_array_row(this->data(), &dim, 1, precision);
        }
        
        
        template<size_t N>
        tensor_index<N> operator+(const tensor_index<N>& A, const tensor_index<N>& B){
            tensor_index<N> C;
            detail::add(A.data(), B.data(), C.data(), N);
            return C;
        }
        
        template<size_t N>
        tensor_index<N> operator-(const tensor_index<N>& A, const tensor_index<N>& B){
            tensor_index<N> C;
            detail::sub(A.data(), B.data(), C.data(), N);
            return C;
        }
        
        
        template <size_t N>
        void circular_shift_index(const tensor_index<N> & Aold,
                                  tensor_index<N> & Anew, int shift)
        {
            size_t rest;
            if (shift >= 0) {
                shift%=N;
                rest = N-shift;
                std::copy(Aold.begin(), Aold.begin()+shift, Anew.begin()+rest);
                std::copy(Aold.begin()+shift, Aold.end(), Anew.begin());
            }
            else{
                shift = -shift;
                shift%=N;
                rest = N-shift;
                std::copy(Aold.begin(), Aold.begin()+rest, Anew.begin()+shift);
                std::copy(Aold.begin()+rest, Aold.end(), Anew.begin());
            }
        }
        
        
        template <size_t N, size_t N1>
        bool reducable_core(const tensor_index<N>& dim, const tensor_index<N1>& reduced)
        {
            bool rr = true;
            if (tensor::detail::prod(reduced.data(), N1) != tensor::detail::prod(dim.data(), N)) {
                rr = false;
            }
            else{
                typename tensor_index<N>::const_iterator itr_start = dim.cbegin(), itr_end = dim.cend(), itr_in = itr_start;
                typename tensor_index<N1>::const_iterator ritr_start = reduced.cbegin(), ritr_end = reduced.cend();
                while (ritr_start != ritr_end) {
                    itr_in = std::find(itr_start, itr_end, *ritr_start);
                    if (*itr_in != 1 && itr_in == itr_end) {
                        rr = false;
                        break;
                    }
                    itr_start = itr_in;
                    ++ritr_start;
                }
            }
            return rr;
        }
        
        template <size_t N, size_t N1>
        bool reducable(const tensor_index<N>& dim, const tensor_index<N1>& reduced)
        {
            return reducable_core(dim, reduced)||reducable_core(reduced, dim);
        }
        
        
        template<typename T>
        struct Index_size{};
        
        template <size_t N>
        struct Index_size<tensor_index<N> > : boost::mpl::int_<N> {};
        
        
        
        
        
        
        
    }
}

#endif /* tensor_index_h */
