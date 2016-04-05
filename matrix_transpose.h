//
//  matrix_transpose.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_matrix_transpose_h
#define guochu_gtensor_matrix_transpose_h


#include <complex>
#if defined BOOST_NUMERIC_BINDINGS_MKL
#include <mkl_trans.h>
#endif
#include "tensor_label.h"


namespace guochu {
    namespace bindings{
        namespace mkl{
            
#ifdef BOOST_NUMERIC_BINDINGS_MKL
            inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, const double alpha, const double *A, size_t lda, double *B, size_t ldb)
            {
                MKL_Domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
            }
            
            inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<double> alpha, const std::complex<double> *A,  size_t lda, std::complex<double> *B, size_t ldb)
            {
                
                MKL_Zomatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
            }
            
            inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<float> alpha, const std::complex<float> *A,  size_t lda, std::complex<float> *B, size_t ldb)
            {
                MKL_Comatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
            }
            
            inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, const float alpha, const float *A,  size_t lda, float *B, size_t ldb)
            {
                MKL_Somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
            }
            
            inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, const double alpha, double *A, size_t lda, size_t ldb)
            {
                MKL_Dimatcopy(ordering, trans, rows, cols, alpha, A, lda, ldb);
            }
            
            inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<double> alpha, std::complex<double> *A, size_t lda, size_t ldb)
            {
                MKL_Zimatcopy(ordering, trans, rows, cols, alpha, A, lda, ldb);
            }
            
            inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<float> alpha, std::complex<float> *A, size_t lda, size_t ldb)
            {
                MKL_Cimatcopy(ordering, trans, rows, cols, alpha, A, lda, ldb);
            }
            
            inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, const float alpha, float *A, size_t lda, size_t ldb)
            {
                MKL_Simatcopy(ordering, trans, rows, cols, alpha, A, lda, ldb);
            }
            
#endif
            
            
        }
        
    }
}

namespace guochu {
    
    namespace tensor {
        
        template<typename T, typename Maj, typename Tr>
        inline void omatcopy(Maj maj, Tr tran, size_t size1, size_t size2, T alpha, const T* A, size_t leada, T* B, size_t leadb)
        {
            BOOST_STATIC_ASSERT(std::is_same<Maj, label::column_major>::value || std::is_same<Maj, label::row_major>::value);
            BOOST_STATIC_ASSERT(std::is_same<Tr, label::no_transpose>::value ||
                                std::is_same<Tr, label::transpose>::value || std::is_same<Tr, label::conjugate>::value);
            return guochu::bindings::mkl::omatcopy(label::tensor_option<Maj>::value, label::tensor_option<Tr>::value, size1, size2, alpha, A, leada, B, leadb);
        }
        
        template<typename T, typename Maj, typename Tr>
        inline void omatcopy(Maj maj, Tr tran, size_t size1, size_t size2, T alpha, const T* A, T* B)
        {
            BOOST_STATIC_ASSERT(std::is_same<Maj, label::column_major>::value || std::is_same<Maj, label::row_major>::value);
            BOOST_STATIC_ASSERT(std::is_same<Tr, label::no_transpose>::value ||
                                std::is_same<Tr, label::transpose>::value || std::is_same<Tr, label::conjugate>::value);
            size_t la = std::is_same<Maj, label::column_major>::value?size1:size2;
            size_t lb = std::is_same<Maj, label::column_major>::value?size2:size1;
            return guochu::bindings::mkl::omatcopy(label::tensor_option<Maj>::value, label::tensor_option<Tr>::value, size1, size2, alpha, A, la, B, lb);
        }
        
        
        /**********************************************************************************
         this function shift the element of the vector forward by "shift" element. for exa-
         mple, Aold=[14,6,7,8,9], shift = 2, then Anew=[7,8,9,14,6]. in case shift > 0
         this function shift the element of the vector backward by "shift" element. for exa-
         mple, Aold=[14,6,7,8,9], shift = 2, then Anew=[8,9,14,6,7]. in case shift < 0
         *********************************************************************************/
        
        template<typename T, typename major, typename Trans>
        void circular_shift_mult_array(const T* A,
                                       const size_t* Adim, major maj, Trans tr, size_t sizeA, T* B, T alpha, int N, int shift)
        {
            size_t size1 = 1, size2 = 1, i, la, lb;
            if (shift >= 0) {
                for (i=0; i<shift; ++i) {size1*=Adim[i];}
                for (i=shift; i<N; ++i) {size2*=Adim[i];}
            }
            else{
                shift = -shift;
                for (i=N-shift; i<N; ++i) {size2*=Adim[i];}
                for (i=0; i<N-shift; ++i) {size1*=Adim[i];}
            }
            la = std::is_same<major, label::column_major>::value?size1:size2;
            lb = std::is_same<major, label::column_major>::value?size2:size1;
            return omatcopy(maj, tr, size1, size2, alpha, A, la, B, lb);
        }
        
    }
}


#endif /* matrix_transpose_h */
