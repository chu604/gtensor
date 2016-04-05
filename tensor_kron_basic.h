//
//  tensor_kron_basic.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_kron_basic_h
#define guochu_gtensor_tensor_kron_basic_h


#include "tensor_label.h"

namespace guochu {
    namespace tensor {
        namespace detail{
            
            // basic kron operation.
            
            template<typename T>
            void kron_basic(T alpha, const T* dataA, size_t sizeA1, size_t sizeA2, label::no_transpose, const T* dataB, size_t sizeB1, size_t sizeB2, T beta, label::no_transpose, T* dataC, label:: column_major)
            {
                size_t itrA1, itrB1;
                size_t itrA2, itrB2;
                size_t temp1, temp2, sizeC1 = sizeA1*sizeB1, interC1, interC2;
                const T *Apoint, *Bpoint;
                T *Cpoint;
                T valueA, valueB;
                if (alpha == 1.0) {
                    if (beta == 0.0) {
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = Apoint[itrA1];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = Apoint[itrA1];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB + beta*Cpoint[temp1];
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    if (beta == 0.0) {
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = alpha*Apoint[itrA1];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = alpha*Apoint[itrA1];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB + beta*Cpoint[temp1];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            
            
            template<typename T>
            void kron_basic(T alpha, const T* dataA, size_t sizeA1, size_t sizeA2, label::no_transpose, const T* dataB, size_t sizeB1, size_t sizeB2, T beta, label::no_transpose, T* dataC, label:: row_major)
            {
                size_t itrA1, itrB1;
                size_t itrA2, itrB2;
                size_t temp1, temp2, sizeC2 = sizeA2*sizeB2, interC1, interC2;
                const T *Apoint, *Bpoint;
                T *Cpoint;
                T valueA, valueB;
                if (alpha == 1.0) {
                    if (beta == 0.0) {
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = Apoint[itrA2];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    temp1=interC1+itrB1;
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    Cpoint = &dataC[temp1*sizeC2];
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        temp2=interC2+itrB2;
                                        valueB = Bpoint[itrB2];
                                        Cpoint[temp2] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = Apoint[itrA2];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    temp1=interC1+itrB1;
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    Cpoint = &dataC[temp1*sizeC2];
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        temp2=interC2+itrB2;
                                        valueB = Bpoint[itrB2];
                                        Cpoint[temp2] = valueA*valueB + beta*Cpoint[temp2];
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    if (beta == 0.0) {
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = alpha*Apoint[itrA2];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    temp1=interC1+itrB1;
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    Cpoint = &dataC[temp1*sizeC2];
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        temp2=interC2+itrB2;
                                        valueB = Bpoint[itrB2];
                                        Cpoint[temp2] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = alpha*Apoint[itrA2];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    temp1=interC1+itrB1;
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    Cpoint = &dataC[temp1*sizeC2];
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        temp2=interC2+itrB2;
                                        valueB = Bpoint[itrB2];
                                        Cpoint[temp2] = valueA*valueB + beta*Cpoint[temp2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            
            
            
            
            // assume A is being transposed, sizeA1 will map to the sizeA2 of the original matrix, and sizeA2 will map to size1 of the original matrix
            template<typename T>
            void kron_basic(T alpha, const T* dataA, size_t sizeA1, size_t sizeA2, label::transpose, const T* dataB, size_t sizeB1, size_t sizeB2, T beta, label::no_transpose, T* dataC, label:: column_major)
            {
                size_t itrA1, itrB1;
                size_t itrA2, itrB2;
                size_t temp1, temp2, sizeC1 = sizeA1*sizeB1, interC1, interC2;
                const T *Apoint, *Bpoint;
                T *Cpoint;
                T valueA, valueB;
                if (alpha == 1.0) {
                    if (beta == 0.0) {
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = Apoint[itrA2];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = Apoint[itrA2];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB + beta*Cpoint[temp1];
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    if (beta == 0.0) {
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = alpha*Apoint[itrA2];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = alpha*Apoint[itrA2];
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB + beta*Cpoint[temp1];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            
            // assume A is being complex conjugated, sizeA1 will map to the sizeA2 of the original matrix, and sizeA2 will map to size1 of the original matrix
            template<typename T>
            void kron_basic(T alpha, const T* dataA, size_t sizeA1, size_t sizeA2, label::conjugate, const T* dataB, size_t sizeB1, size_t sizeB2, T beta, label::no_transpose, T* dataC, label:: column_major)
            {
                size_t itrA1, itrB1;
                size_t itrA2, itrB2;
                size_t temp1, temp2, sizeC1 = sizeA1*sizeB1, interC1, interC2;
                const T *Apoint, *Bpoint;
                T *Cpoint;
                T valueA, valueB;
                if (alpha == 1.0) {
                    if (beta == 0.0) {
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = conj(Apoint[itrA2]);
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = conj(Apoint[itrA2]);
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB + beta*Cpoint[temp1];
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    if (beta == 0.0) {
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = alpha*conj(Apoint[itrA2]);
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                            Apoint = &dataA[itrA1*sizeA2];
                            interC1 = itrA1*sizeB1;
                            for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                                interC2 = itrA2*sizeB2;
                                valueA = alpha*conj(Apoint[itrA2]);
                                for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                    Bpoint = &dataB[itrB2*sizeB1];
                                    temp2=interC2+itrB2;
                                    Cpoint = &dataC[temp2*sizeC1];
                                    for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                        valueB = Bpoint[itrB1];
                                        temp1=interC1+itrB1;
                                        Cpoint[temp1] = valueA*valueB + beta*Cpoint[temp1];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            
            // assume B is being transposed
            template<typename T>
            void kron_basic(T alpha, const T* dataA, size_t sizeA1, size_t sizeA2, label::no_transpose, const T* dataB, size_t sizeB1, size_t sizeB2, T beta, label::transpose, T* dataC, label:: column_major)
            {
                size_t itrA1, itrB1;
                size_t itrA2, itrB2;
                size_t temp1, temp2, sizeC1 = sizeA1*sizeB1, interC1, interC2;
                const T *Apoint, *Bpoint;
                T *Cpoint;
                T valueA, valueB;
                if (alpha == 1.0) {
                    if (beta == 0.0) {
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = Bpoint[itrB2];
                                        temp2=interC2+itrB2;
                                        dataC[temp2*sizeC1+temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = Bpoint[itrB2];
                                        temp2=interC2+itrB2;
                                        Cpoint = &dataC[temp2*sizeC1+temp1];
                                        *Cpoint = valueA*valueB + beta*(*Cpoint);
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    if (beta == 0.0) {
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = alpha*Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = Bpoint[itrB2];
                                        temp2=interC2+itrB2;
                                        dataC[temp2*sizeC1+temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = alpha*Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = Bpoint[itrB2];
                                        temp2=interC2+itrB2;
                                        Cpoint = &dataC[temp2*sizeC1+temp1];
                                        *Cpoint = valueA*valueB + beta*(*Cpoint);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            
            // assume B is being conjugate transposed
            template<typename T>
            void kron_basic(T alpha, const T* dataA, size_t sizeA1, size_t sizeA2, label::no_transpose, const T* dataB, size_t sizeB1, size_t sizeB2, T beta, label::conjugate, T* dataC, label:: column_major)
            {
                size_t itrA1, itrB1;
                size_t itrA2, itrB2;
                size_t temp1, temp2, sizeC1 = sizeA1*sizeB1, interC1, interC2;
                const T *Apoint, *Bpoint;
                T *Cpoint;
                T valueA, valueB;
                if (alpha == 1.0) {
                    if (beta == 0.0) {
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = conj(Bpoint[itrB2]);
                                        temp2=interC2+itrB2;
                                        dataC[temp2*sizeC1+temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = conj(Bpoint[itrB2]);
                                        temp2=interC2+itrB2;
                                        Cpoint = &dataC[temp2*sizeC1+temp1];
                                        *Cpoint = valueA*valueB + beta*(*Cpoint);
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    if (beta == 0.0) {
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = alpha*Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = conj(Bpoint[itrB2]);
                                        temp2=interC2+itrB2;
                                        dataC[temp2*sizeC1+temp1] = valueA*valueB;
                                    }
                                }
                            }
                        }
                    }
                    else{
                        for (itrA2=0; itrA2 < sizeA2; ++itrA2) {
                            Apoint = &dataA[itrA2*sizeA1];
                            interC2 = itrA2*sizeB2;
                            for (itrA1=0; itrA1<sizeA1; ++itrA1) {
                                interC1 = itrA1*sizeB1;
                                valueA = alpha*Apoint[itrA1];
                                for (itrB1=0; itrB1<sizeB1; ++itrB1) {
                                    Bpoint = &dataB[itrB1*sizeB2];
                                    temp1=interC1+itrB1;
                                    for (itrB2=0; itrB2<sizeB2; ++itrB2) {
                                        valueB = conj(Bpoint[itrB2]);
                                        temp2=interC2+itrB2;
                                        Cpoint = &dataC[temp2*sizeC1+temp1];
                                        *Cpoint = valueA*valueB + beta*(*Cpoint);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            
        }
    }
    
}

#endif /* tensor_kron_basic_h */
