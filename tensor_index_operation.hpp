//
//  tensor_index_operation.hpp
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_index_operation_hpp
#define guochu_gtensor_tensor_index_operation_hpp

#include <assert.h>
#include <complex>
#include <vector>
#include <iostream>
#include "tensor_label.h"


namespace guochu {
    
    double conj(double a);
//    std::complex<double> conj(std::complex<double> a);
    double real(double a);
//    double real(std::complex<double> a);
    double imag(double a);
//    double imag(std::complex<double> a);
    
    namespace tensor {
        namespace detail{
            
            
            void conj(const double* a1, double* a2, size_t n);
            void conj(const std::complex<double>* a1, std::complex<double>* a2, size_t n);
            void real(const double* a1, double* a2, size_t n);
            void real(const std::complex<double>* a1, double* a2, size_t n);
            void imag(const double* a1, double* a2, size_t n);
            void imag(const std::complex<double>* a1, double* a2, size_t n);
            
            
            void dim2cudim_col(const size_t* dim, size_t* cudim, size_t N);
            void dim2cudim_row(const size_t* dim, size_t* cudim, size_t N);
            
            size_t mind2sind_col(const size_t* multindex, const size_t* cudim, size_t N); // count from 0
            size_t mind2sind_row(const size_t* multindex, const size_t* cudim, size_t N); // count from 0
            size_t mind2sind_col(const size_t* multindex,
                                 const size_t* startindex,  //count from here
                                 const size_t* cudim,
                                 size_t N);
            size_t mind2sind_row(const size_t* multindex,
                                 const size_t* startindex,  //count from here
                                 const size_t* cudim,
                                 size_t N);
            
            void sind2mind_col(size_t singleindex, size_t* multindex,
                               const size_t* cudim, size_t N);
            
            void sind2mind_row(size_t singleindex, size_t* multindex,
                               const size_t* cudim, size_t N);
            
            
            void sortlist(const size_t* listin,
                          size_t* listout, const size_t* hash, size_t N);
            
            size_t mapsind2sindbyhashcol(const size_t* cudimin, size_t aout,
                                         const size_t* cudimout, const size_t* hash, size_t N);
            
            size_t mapsind2sindbyhashrow(const size_t* cudimin, size_t aout,
                                         const size_t* cudimout, const size_t* hash, size_t N);
            
            size_t mapsind2sindcol2row(const size_t* cudimrow, size_t aout,
                                       const size_t* cudimcol, size_t N);
            
            size_t mapsind2sindrow2col(const size_t* cudimcol, size_t aout,
                                       const size_t* cudimrow, size_t N);
            
            size_t sindview2sind_col(const size_t* cudimall, size_t asubview,
                                     const size_t* cudimasub,
                                     const size_t* startindex, size_t N);
            
            size_t sindview2sind_row(const size_t* cudimall, size_t asubview,
                                     const size_t* cudimasub,
                                     const size_t* startindex, size_t N);
            
            size_t sindview2sind_col(const size_t* cudima, size_t asub,
                                     const size_t* cudimasub, size_t N);
            
            size_t sindview2sind_row(const size_t* cudima, size_t asub,
                                     const size_t* cudimasub, size_t N);
            
            double mean(const double* vec, size_t size);
            
            double stdvar(const double* vec, size_t size);
            
            size_t mind2sind_col(size_t i, size_t j, const size_t* cudim);
            size_t mind2sind_col(size_t i, size_t j, size_t k, const size_t* cudim);
            size_t mind2sind_col(size_t i, size_t j, size_t k, size_t l, const size_t* cudim);
            size_t mind2sind_col(size_t i, size_t j, size_t k, size_t l, size_t m, const size_t* cudim);
            size_t mind2sind_col(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, const size_t* cudim);
            size_t mind2sind_row(size_t i, size_t j, const size_t* cudim);
            size_t mind2sind_row(size_t i, size_t j, size_t k, const size_t* cudim);
            size_t mind2sind_row(size_t i, size_t j, size_t k, size_t l, const size_t* cudim);
            size_t mind2sind_row(size_t i, size_t j, size_t k, size_t l, size_t m, const size_t* cudim);
            size_t mind2sind_row(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, const size_t* cudim);
            
            void circular_shift_index(const size_t* Aold,
                                      size_t* Anew, size_t N, int shift);
            
        }
    }
}

namespace guochu {
    namespace tensor {
        
        inline void dim2cudim(const size_t* dim, size_t* cudim, size_t N, label::column_major)
        {
            return detail::dim2cudim_col(dim, cudim, N);
        }
        
        inline void dim2cudim(const size_t* dim, size_t* cudim, size_t N, label::row_major)
        {
            return detail::dim2cudim_row(dim, cudim, N);
        }
        
        inline size_t mind2sind(size_t i, size_t j, const size_t* cudim, label::column_major)
        {
            return detail::mind2sind_col(i, j, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, const size_t* cudim, label::column_major)
        {
            return detail::mind2sind_col(i, j, k, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, const size_t* cudim, label::column_major)
        {
            return detail::mind2sind_col(i, j, k, l, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, const size_t* cudim, label::column_major)
        {
            return detail::mind2sind_col(i, j, k, l, m, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, const size_t* cudim, label::column_major)
        {
            return detail::mind2sind_col(i, j, k, l, m, n, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, const size_t* cudim, label::row_major)
        {
            return detail::mind2sind_row(i, j, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, const size_t* cudim, label::row_major)
        {
            return detail::mind2sind_row(i, j, k, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, const size_t* cudim, label::row_major)
        {
            return detail::mind2sind_row(i, j, k, l, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, const size_t* cudim, label::row_major)
        {
            return detail::mind2sind_row(i, j, k, l, m, cudim);
        }
        
        inline size_t mind2sind(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, const size_t* cudim, label::row_major)
        {
            return detail::mind2sind_row(i, j, k, l, m, n, cudim);
        }
        
        
        
        
        inline size_t mind2sind(const size_t* multindex, const size_t* cudim, size_t N, label::column_major)
        {
            return detail::mind2sind_col(multindex, cudim, N);
        }
        
        inline size_t mind2sind(const size_t* multindex, const size_t* cudim, size_t N, label::row_major)
        {
            return detail::mind2sind_row(multindex, cudim, N);
        }
        
        inline size_t mind2sind(const size_t* multindex,
                                const size_t* startindex,
                                const size_t* cudim,
                                size_t N, label::column_major)
        {
            return detail::mind2sind_col(multindex, startindex, cudim, N);
        }
        
        inline size_t mind2sind(const size_t* multindex,
                                const size_t* startindex,
                                const size_t* cudim,
                                size_t N, label::row_major)
        {
            return detail::mind2sind_row(multindex, startindex, cudim, N);
        }
        
        inline void sind2mind(size_t singleindex, size_t* multindex,
                              const size_t* cudim, size_t N, label::column_major)
        {
            return detail::sind2mind_col(singleindex, multindex, cudim, N);
        }
        
        inline void sind2mind(size_t singleindex, size_t* multindex,
                              const size_t* cudim, size_t N, label::row_major)
        {
            return detail::sind2mind_row(singleindex, multindex, cudim, N);
        }
        
        inline size_t mapsind2sindbyhash(const size_t* cudimin, size_t aout,
                                         const size_t* cudimout, const size_t* hash,
                                         size_t N, label::column_major)
        {
            return detail::mapsind2sindbyhashcol(cudimin, aout, cudimout, hash, N);
        }
        
        inline size_t mapsind2sindbyhash(const size_t* cudimin, size_t aout,
                                         const size_t* cudimout, const size_t* hash,
                                         size_t N, label::row_major)
        {
            return detail::mapsind2sindbyhashrow(cudimin, aout, cudimout, hash, N);
        }
        
        inline size_t sindview2sind(const size_t* cudimall, size_t asubview,
                                    const size_t* cudimasub,
                                    const size_t* startindex, size_t N, label::column_major)
        {
            return detail::sindview2sind_col(cudimall, asubview, cudimasub, startindex, N);
        }
        
        inline size_t sindview2sind(const size_t* cudimall, size_t asubview,
                                    const size_t* cudimasub,
                                    const size_t* startindex, size_t N, label::row_major)
        {
            return detail::sindview2sind_row(cudimall, asubview, cudimasub, startindex, N);
        }
        
        inline size_t sindview2sind(const size_t* cudima, size_t asub,
                                    const size_t* cudimasub, size_t N, label::column_major)
        {
            return detail::sindview2sind_col(cudima, asub, cudimasub, N);
        }
        
        inline size_t sindview2sind(const size_t* cudima, size_t asub,
                                    const size_t* cudimasub, size_t N, label::row_major)
        {
            return detail::sindview2sind_row(cudima, asub, cudimasub, N);
        }
        
        
        
    }
}


#endif /* tensor_index_operation_hpp */
