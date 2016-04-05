
//
//  tensor_utility.h
//  gtensor
//
//  Created by guochu on 19/11/15.
//  Copyright © 2015 guochu. All rights reserved.
//

#ifndef guochu_gtensor_gtensor_utility_h
#define guochu_gtensor_gtensor_utility_h

#include "../config.h"
#include <assert.h>
#include <boost/static_assert.hpp>
#include <array>
#include <boost/numeric/bindings/blas/detail/blas_option.hpp>
#include <vector>
#include "tensor_label.h"
#include "tensor_index_operation.hpp"



namespace guochu {
    
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, std::valarray<T> >::type
    real(const std::valarray<std::complex<T> > & vec)
    {
        std::valarray<T> vecreal(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            vecreal[i] = std::real(vec[i]);
        }
        return vecreal;
    }
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, std::valarray<T> >::type
    real(const std::valarray<T> & vec)
    {
        std::valarray<T> vecreal(vec);
        return vecreal;
    }
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, std::valarray<T> >::type
    imag(const std::valarray<std::complex<T> > & vec)
    {
        std::valarray<T> vecreal(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            vecreal[i] = std::imag(vec[i]);
        }
        return vecreal;
    }
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, std::valarray<T> >::type
    imag(const std::valarray<T> & vec)
    {
        std::valarray<T> vecreal(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            vecreal[i] = 0.;
        }
        return vecreal;
    }
    
    
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, std::valarray<T> >::type
    conj(const std::valarray<T> & vec)
    {
        std::valarray<T> vec1(vec);
        return vec1;
    }
    
    template<typename T>
    typename std::enable_if<boost::is_complex<T>::value, std::valarray<T> >::type
    conj(const std::valarray<T> & vec)
    {
        std::valarray<T> vec1(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            vec1[i] = std::conj(vec[i]);
        }
        return vec1;
    }
    
    
    template<typename T>
    T sum(const std::valarray<T> & vec)
    {
        T ss = 0.;
        for (size_t i = 0; i < vec.size(); ++i) {
            ss += vec[i];
        }
        return ss;
    }
    
}



namespace guochu {
    
    namespace tensor{
        namespace detail{
            
            template<typename T>
            T sum(const T* a, size_t n)
            {
                size_t i = 0;
                T ss = 0.;
                while (i < n) {
                    ss += *a;
                    ++i;
                    ++a;
                }
                return ss;
            };
            
            
            template<typename T>
            void min(const T* A,
                     const T* B, T* minvec, size_t N)
            {
                size_t i = 0;
                while (i++ < N) {
                    *minvec++ = std::min(*A++, *B++);
                }
            }
            
            template<typename T>
            void max(const T* A,
                     const T* B, size_t* minvec, size_t N)
            {
                size_t i = 0;
                while (i++ < N) {
                    *minvec++ = std::max(*A++, *B++);
                }
            }
            
            template<typename T>
            T prod(const T* A, size_t N)
            {
                if (N == 0) {
                    return 0;
                }
                size_t n = 1, i = 0;
                while (i++ < N) {
                    n *= (*A++);
                }
                return n;
            }
            
            template<typename T>
            void add(const T* A,
                     const T* B, T* C, size_t N)
            {
                size_t i = 0;
                while (i++ < N) {
                    *C++ = *A++ + *B++;
                }
            }
            
            template<typename T>
            void sub(const T* A,
                     const T* B, T* C, size_t N)
            {
                size_t i = 0;
                while (i++ < N) {
                    *C++ = *A++ - *B++;
                }
            }
            
            template<typename TA, typename TB>
            void permute_mult_array_col(const TA* dataA, const size_t* cudimA,
                                        size_t Asize, TB* dataB,
                                        const size_t* cudimB, const size_t* hash, size_t N) // B is column major, tested!
            {
                size_t i = 0, mapi = mapsind2sindbyhashcol(cudimA, i, cudimB, hash, N);
                *dataB = dataA[mapi];
                while (++i < Asize) {
                    mapi = mapsind2sindbyhashcol(cudimA, i, cudimB, hash, N);
                    assert(mapi < Asize);
                    *(++dataB) = dataA[mapi];
                }
            }
            
            template<typename TA, typename TB>
            void permute_mult_array_row(const TA* dataA,
                                        const size_t* cudimA, size_t Asize,
                                        TB* dataB, const size_t* cudimB,
                                        const size_t* hash, size_t N)                     // B is row major, tested!
            {
                size_t i = 0, mapi = mapsind2sindbyhashrow(cudimA, i, cudimB, hash, N);
                *dataB = dataA[mapi];
                while (++i < Asize) {
                    mapi = mapsind2sindbyhashrow(cudimA, i, cudimB, hash, N);
                    assert(mapi < Asize);
                    *(++dataB) = dataA[mapi];
                }
            }
            
            template<typename TA, typename TB>
            void copyd2d(const TA* dataA,
                         const size_t* cudimA, TB* dataB,
                         const size_t* cudimB,
                         size_t size, size_t N) //tested!
            {
                if (size > 0) {
                    size_t i = 0;
                    *dataB = *dataA;
                    while (++i < size) {
                        *(++dataB) = *(++dataA);
                    }
                }
            }
            
            template<typename TA, typename TB>
            void copyd2d_col2row(const TA* dataA,
                                 const size_t* cudimA,
                                 TB* dataB,
                                 const size_t* cudimB, size_t size, size_t N) //tested!
            {
                if (size > 0) {
                    size_t i = 0, mapi = mapsind2sindcol2row(cudimB, i, cudimA, N);
                    dataB[mapi] = *dataA;
                    while (++i < size) {
                        mapi = mapsind2sindcol2row(cudimB, i, cudimA, N);
                        dataB[mapi] = *(++dataA);
                    }
                }
            }
            
            template<typename TA, typename TB>
            void copyd2d_row2col(const TA* dataA,
                                 const size_t* cudimA,
                                 TB* dataB,
                                 const size_t* cudimB,
                                 size_t size, size_t N)    //tested!
            {
                if (size > 0) {
                    size_t i = 0, mapi = mapsind2sindrow2col(cudimB, i, cudimA, N);
                    dataB[mapi] = *dataA;
                    while (++i < size) {
                        mapi = mapsind2sindrow2col(cudimB, i, cudimA, N);
                        dataB[mapi] = *(++dataA);
                    }
                }
            }
            
            template<typename T>
            void print_mult_array_row(const T* data,
                                      const size_t* dim,
                                      size_t N,
                                      size_t precision)  //tested!
            {
                if (N == 0) {
                    return;
                }
                std::cout.precision(precision);
                if (N == 1) {
                    std::cout << "[ ";
                    for (size_t i = 0; i < dim[0]; ++i) {
                        std::cout << data[i] << ",";
                    }
                    std::cout << " ]";
                }
                else{
                    size_t size1 = 1, i;
                    std::vector<size_t> dim1(N-1);
                    for (i = 1; i < N; ++i) {
                        dim1[i-1] = dim[i];
                        size1 *= dim[i];
                    }
                    const T* data1;
                    std::cout << "[ ";
                    for (i = 0; i < dim[0]; ++i) {
                        data1 = &data[i*size1];
                        print_mult_array_row(data1, dim1.data(), dim1.size(), precision);
                        if (i != dim[0]-1) {
                            std::cout <<", ";
                        }
                    }
                    std::cout << " ]";
                }
                std::cout << std::endl;
            }
            
            
            template<typename T>
            void print_mult_array_col(const T* data,
                                      const size_t* dim,
                                      size_t N,
                                      size_t precision)   //tested!
            {
                if (N == 0) {
                    return;
                }
                size_t totalsize = prod(dim, N);
                std::vector<T> datarow(totalsize);
                std::vector<size_t> cudimA(N), cudimB(N);
                dim2cudim_col(dim, cudimA.data(), N);
                dim2cudim_row(dim, cudimB.data(), N);
                copyd2d_col2row(data, cudimA.data(), datarow.data(), cudimB.data(), totalsize, N);
                print_mult_array_row(datarow.data(), dim, N, precision);
            }
            
            
            //implement colmajored matrix multiply with a diagonal matrix which is stored by one dimensional vector.
            //start is the first position of the input matrix, size1 and size2 are the first and second dimension of the matrix, vec is the first position of the vector.
            template<typename TA, typename TB>
            void general_rowscale_col(TA *start,
                                      size_t size1, size_t size2,
                                      const TB *vec)
            {
                if (size1*size2 != 0) {
                    size_t i, j, k = size1-1;
                    TA *value;
                    const TB *vecr;
                    for (i = 0; i < size2; ++i) {
                        value = &start[i*size1];
                        vecr = vec;
                        j = 0;
                        while (j++ < k) {
                            *value++ *= *vecr++;
                        }
                        *value *= *vecr;
                    }
                }
            }
            
            template<typename TA, typename TB>
            void general_colscale_col(TA *start,
                                      size_t size1, size_t size2,
                                      const TB *vec)
            {
                if (size1*size2 != 0) {
                    size_t i = 0, j, k = size1-1;
                    TA *value;
                    TB scal;
                    for (i = 0; i < size2; ++i) {
                        value = &start[i*size1];
                        scal = vec[i];
                        j = 0;
                        while (j++ < k) {
                            *value++ *= scal;
                        }
                        *value *= scal;
                    }
                }
            }
            
            
        }
    }
}



namespace guochu { namespace tensor {
    
    /**************************************************************************************
     auxiliary functions of index operation
     *************************************************************************************/
    
    /**************************************************************************************
     the input is the dimension of the tensor, the output is the cumulate dimension of the
     tensor. for example, a tensor with dimension dim = [4,3,5], then the output cudim =
     [1, 4, 12]. This is used for colmajor storage.
     *************************************************************************************/
    
    template <size_t N>
    inline void dim2cudim(const std::array<size_t, N> & dim, std::array<size_t, N> & cudim, label::column_major maj)
    {
        return tensor::dim2cudim(dim.data(), cudim.data(), N, maj);
    }

    template <size_t N>
    inline void dim2cudim(const std::array<size_t, N> & dim, std::array<size_t, N> & cudim, label::row_major maj)
    {
        return tensor::dim2cudim(dim.data(), cudim.data(), N, maj);
    }

    
    /***************************************************************************************
     the input of this function is the vector multindex, which is the dimension of the tensor
     , the out put singleindex is the index if we look at the tensor as a column vector. for
     example, cudim is needed as the cumulate dimension of the tensor. for example, for a t-
     ensor with dim = [4,3,5], then cudim should be [1,4,12]. If multiindex = [2,2,3], then
     singleindex = 2*1+2*4+3*12 = 46.
     **************************************************************************************/
    
    template <size_t N>
    size_t mind2sind(const std::array<size_t, N> & multindex, const std::array<size_t, N> & cudim, label::column_major maj)
    {
        return tensor::mind2sind(multindex.data(), cudim.data(), N, maj);
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, const std::array<size_t, N> & cudim, label::column_major colmaj){
        BOOST_STATIC_ASSERT(N==2);
        return i + j*cudim[1];
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, const std::array<size_t, N> & cudim, label::column_major colmaj){
        BOOST_STATIC_ASSERT(N==3);
        return i + j*cudim[1] + k*cudim[2];
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, const std::array<size_t, N> & cudim, label::column_major colmaj){
        BOOST_STATIC_ASSERT(N==4);
        return i + j*cudim[1] + k*cudim[2] + l*cudim[3];
    }

    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, const std::array<size_t, N> & cudim, label::column_major colmaj){
        BOOST_STATIC_ASSERT(N==5);
        return i + j*cudim[1] + k*cudim[2] + l*cudim[3] + m*cudim[4];
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, const std::array<size_t, N> & cudim, label::column_major colmaj){
        BOOST_STATIC_ASSERT(N==6);
        return i + j*cudim[1] + k*cudim[2] + l*cudim[3] + m*cudim[4] + n*cudim[5];
    }

    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, const std::array<size_t, N> & cudim, label::row_major rowmaj){
        BOOST_STATIC_ASSERT(N==2);
        return i*cudim[0] + j;
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, const std::array<size_t, N> & cudim, label::row_major rowmaj){
        BOOST_STATIC_ASSERT(N==3);
        return i*cudim[0] + j*cudim[1] + k;
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, const std::array<size_t, N> & cudim, label::row_major rowmaj){
        BOOST_STATIC_ASSERT(N==4);
        return i*cudim[0] + j*cudim[1] + k*cudim[2] + l;
    }

    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, const std::array<size_t, N> & cudim, label::row_major rowmaj){
        BOOST_STATIC_ASSERT(N==5);
        return i*cudim[0] + j*cudim[1] + k*cudim[2] + l*cudim[3] + m;
    }
    
    template<size_t N>
    inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, const std::array<size_t, N> & cudim, label::row_major rowmaj){
        BOOST_STATIC_ASSERT(N==6);
        return i*cudim[0] + j*cudim[1] + k*cudim[2] + l*cudim[3] + m*cudim[4] + n;
    }
    
    template <size_t N>
    size_t mind2sind(const std::array<size_t, N> & multindex, const std::array<size_t, N> & cudim, label::row_major maj)
    {
        return tensor::mind2sind(multindex.data(), cudim.data(), N, maj);
    }
    
    template <size_t N>
    size_t mind2sind(const std::array<size_t, N> & multindex,
                   const std::array<size_t, N> & startindex,
                   const std::array<size_t, N> & cudim,
                   label::column_major maj)
    {
        return tensor::mind2sind(multindex.data(), startindex.data(), cudim.data(), N, maj);
    }
    
    template <size_t N>
    size_t mind2sind(const std::array<size_t, N> & multindex,
                   const std::array<size_t, N> & startindex,
                   const std::array<size_t, N> & cudim, label::row_major maj)
    {
        return tensor::mind2sind(multindex.data(), startindex.data(), cudim.data(), N, maj);
    }


    /***************************************************************************************
     the input of this function is the singleindex, which is the dimension of the tensor if
     it is viewed as a column vector. the out put multiindex is the index if we view the te-
     nsor really as a multidimensional tensor. cudim is needed as the cumulate dimension of
     the tensor. for example, for a tensor with dim = [4,3,5], then cudim should be [1,4,12]
     . If singleindex = 46, then multiindex= [((46%12)%4)/1, (46%12)/4, 46/12] =[2,2,3] .
     **************************************************************************************/

    
    template <size_t N>
    void sind2mind(size_t singleindex, std::array<size_t, N> & multindex,
                   const std::array<size_t, N> & cudim, label::column_major maj)
    {
        return tensor::sind2mind(singleindex, multindex.data(), cudim.data(), N, maj);
    }
    
    template <size_t N>
    void sind2mind(size_t singleindex, std::array<size_t, N> & multindex,
                   const std::array<size_t, N> & cudim, label::row_major maj)
    {
        return tensor::sind2mind(singleindex, multindex.data(), cudim.data(), N, maj);
    }

    
    /***********************************************************************************
     this function sort the listin according to the table hash. For example, if hash =
     [2,3,1,4,5], and listin = [7,10,19,12,8], then listout = [10,19,7,12,8].
     **********************************************************************************/
    
    template <size_t N>
    void sortlist(const std::array<size_t, N> & listin,
                  std::array<size_t, N> & listout, const std::array<size_t, N> & hash)
    {
        tensor::detail::sortlist(listin.data(), listout.data(), hash.data(), N);
    }
    
    template <size_t N>
    size_t sum(const std::array<size_t, N> & A)
    {
        return detail::sum(A.data(), N);
    }
    
    template <size_t N>
    size_t prod(const std::array<size_t, N> & A)
    {
        return detail::prod(A.data(), N);
    }
    
    /***********************************************************************************
     this function is a bit hard to explain...
     for a multidimensional tensor with cudimout, we have a index out, and we have anoth-
     er multidimensioanal which has the same rank, and has the cumulate dimension cudimin
     . the two tensor indexes are related by hash, which is a permutation from 1 to rank.
     size is the rank of two tensors. ain is the out put, which is the map of aout. hash
     maps the index hash[i] into i.
     ***********************************************************************************/
    
    template <size_t N>
    inline size_t mapsind2sindbyhashcol(const std::array<size_t, N> & cudimin, size_t aout,
                                      const std::array<size_t, N> & cudimout, const std::array<size_t, N> & hash)
    {
        return tensor::detail::mapsind2sindbyhashcol(cudimin.data(), aout, cudimout.data(), hash.data(), N);
    }

    
    template <size_t N>
    inline size_t mapsind2sindbyhashrow(const std::array<size_t, N> & cudimin, size_t aout,
                                      const std::array<size_t, N> & cudimout, const std::array<size_t, N> & hash)
    {
        return tensor::detail::mapsind2sindbyhashrow(cudimin.data(), aout, cudimout.data(), hash.data(), N);
    }


    template<typename T1, typename T2, size_t N>
    void permute_mult_array(const T1* dataA, const std::array<size_t, N>& cudimA,
                            size_t Asize, label::column_major, T2* dataB,
                            const std::array<size_t, N>& cudimB, std::array<size_t, N> hashtable)
    {
        return tensor::detail::permute_mult_array_col(dataA, cudimA.data(), Asize, dataB, cudimB.data(), hashtable.data(), N);
    }
    
    
    template<typename T1, typename T2, size_t N>
    void permute_mult_array(const T1* dataA,
                            std::array<size_t, N> cudimA, size_t Asize,
                            label::row_major, T2* dataB, std::array<size_t, N> cudimB,
                            std::array<size_t, N> hashtable)
    {
        return tensor::detail::permute_mult_array_row(dataA, cudimA.data(), Asize, dataB, cudimB.data(), hashtable.data(), N);
    }

    
    template <size_t N>
    inline size_t mapsind2sindc2r(const std::array<size_t, N> & cudimrow, size_t aout,
                                const std::array<size_t, N> & cudimcol)
    {
        return tensor::detail::mapsind2sindcol2row(cudimrow.data(), aout, cudimcol.data(), N);
    }
    
    template <size_t N>
    inline size_t mapsind2sindr2c(const std::array<size_t, N> & cudimcol, size_t aout,
                                const std::array<size_t, N> & cudimrow)
    {
        return tensor::detail::mapsind2sindrow2col(cudimcol.data(), aout, cudimrow.data(), N);
    }

    template <size_t N>
    inline size_t sindview2sind(const std::array<size_t, N> & cudimall, size_t asubview,
                              const std::array<size_t, N> & cudimasub,
                              const std::array<size_t, N> & startindex, label::column_major)
    {
        return tensor::detail::sindview2sind_col(cudimall.data(), asubview, cudimasub.data(), startindex.data(), N);
    }
    
    template <size_t N>
    inline size_t sindview2sind(const std::array<size_t, N> & cudimall, size_t asubview,
                                const std::array<size_t, N> & cudimasub,
                                const std::array<size_t, N> & startindex, label::row_major)
    {
        return tensor::detail::sindview2sind_row(cudimall.data(), asubview, cudimasub.data(), startindex.data(), N);
    }

    template <size_t N>
    inline size_t sindview2sind(const std::array<size_t, N> & cudima, size_t asub,
                              const std::array<size_t, N> & cudimasub, label::column_major)
    {
        return tensor::detail::sindview2sind_col(cudima.data(), asub, cudimasub.data(), N);
    }
    
    template <size_t N>
    inline size_t sindview2sind(const std::array<size_t, N> & cudima, size_t asub,
                              const std::array<size_t, N> & cudimasub, label::row_major)
    {
        return tensor::detail::sindview2sind_row(cudima.data(), asub, cudimasub.data(), N);
    }
    
    template<typename T1, typename T2, size_t N, typename Maj>
    void copyd2d(const T1* dataA,
                 const std::array<size_t, N> & cudimA,
                 Maj, T2* dataB,
                 const std::array<size_t, N> & cudimB,
                 Maj, size_t size)
    {
        return tensor::detail::copyd2d(dataA, cudimA.data(), dataB, cudimB.data(), size, N);
    }
    
    template<typename T1, typename T2, size_t N>
    void copyd2d(const T1* dataA,
                 const std::array<size_t, N> & cudimA,
                 label::column_major, T2* dataB,
                 const std::array<size_t, N> & cudimB, label::row_major, size_t size)
    {
        return tensor::detail::copyd2d_col2row(dataA, cudimA.data(), dataB, cudimB.data(), size, N);
    }
    
    template<typename T1, typename T2, size_t N>
    void copyd2d(const T1* dataA,
                 const std::array<size_t, N> & cudimA,
                 label::row_major, T2* dataB,
                 const std::array<size_t, N> & cudimB,
                 label::column_major, size_t size)
    {
        return tensor::detail::copyd2d_row2col(dataA, cudimA.data(), dataB, cudimB.data(), size, N);
    }
    
    template<typename T1, typename T2>
    void general_rowscale(T1 *start,
                          size_t size1, size_t size2,
                          const T2 *vec, label::column_major)
    {
        return tensor::detail::general_rowscale_col(start, size1, size2, vec);
    }
    
    template<typename T1, typename T2>
    void general_colscale(T1 *start,
                          size_t size1, size_t size2,
                          const T2 *vec, label::column_major)
    {
        return tensor::detail::general_colscale_col(start, size1, size2, vec);
    }


    template<typename T>
    void print_mult_array(const T* data,
                          const size_t* dim, label::row_major,
                          size_t N,
                          size_t precision = 6)
    {
        std::cout << "( ";
        for (size_t i = 0; i < N; ++i) {
            std::cout << dim[i] << ",";
        }
        std::cout << " )\n";
        return detail::print_mult_array_row(data, dim, N, precision);
    }
    
    template<typename T>
    void print_mult_array(const T* data,
                          const size_t* dim, label::column_major,
                          size_t N,
                          size_t precision = 6)
    {
        std::cout << "( ";
        for (size_t i = 0; i < N; ++i) {
            std::cout << dim[i] << ",";
        }
        std::cout << " )\n";
        return detail::print_mult_array_col(data, dim, N, precision);
    }
    
    template<typename T>
    void print_vector(const T* data, size_t size)
    {
        std::cout << "[ ";
        for (size_t i = 0; i < size; ++i) {
            std::cout << data[i] << ",";
        }
        std::cout << " ]" << std::endl;
        
    }
    
    
    template <typename T>
    T mean(const T* vec, size_t size)
    {
        T meanvalue = 0.0;
        for (size_t i = 0; i<size; ++i) {
            meanvalue += vec[i];
        }
        meanvalue/=size;
        return meanvalue;
    }
    template <typename T>
    T stdvar(const T* vec, size_t size)
    {
        T meanvalue = mean(vec, size), value=0.0;
        for (size_t i=0; i < size; ++i) {
            value+=(vec[i]-meanvalue)*(vec[i]-meanvalue);
        }
        return value;
    }
    

    
}
}




#endif /* tensor_utility_h */


