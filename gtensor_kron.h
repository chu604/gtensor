


#ifndef guochu_gtensor_gtensor_kron_h
#define guochu_gtensor_gtensor_kron_h

#include "gtensor.h"
#include "../triple.h"
#include "tensor_kron_basic.h"

namespace guochu {

    template <typename TensorA, typename TensorB, typename TensorC, typename T>
    void kron(T alpha, const tensor_expression<TensorA> & A, const tensor_expression<TensorB> & B, T beta, tensor_expression<TensorC> & C)
    {
        BOOST_STATIC_ASSERT(std::is_same<typename TensorA::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::is_same<typename TensorB::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::is_same<typename TensorC::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::tuple_size<typename TensorA::dimvector_type>::value == 2);
        BOOST_STATIC_ASSERT(std::tuple_size<typename TensorB::dimvector_type>::value == 2);
        BOOST_STATIC_ASSERT(std::tuple_size<typename TensorC::dimvector_type>::value == 2);
        size_t sizeA1 = A().size1(), sizeA2 = A().size2();
        size_t sizeB1 = B().size1(), sizeB2 = B().size2();
        if (C().size1() != sizeA1*sizeB1) {
            throw std::runtime_error("kron(A,B)=C, the first size does not match.\n");
        }
        if (C().size2() != sizeA2*sizeB2) {
            throw std::runtime_error("kron(A,B)=C, the second size does not match.\n");
        }
        size_t itrA1, itrB1;
        size_t itrA2, itrB2;
        size_t temp1, temp2, temp;
        temp = sizeA1*sizeB1;
        typename TensorA::value_type valueA, valueB;
        if (beta==0.) {
            if (alpha==1.) {
                for (itrA2=0; itrA2<sizeA2; itrA2++) {
                    for (itrA1=0; itrA1<sizeA1; itrA1++) {
                        valueA = A()({itrA1, itrA2});
                        for (itrB2=0; itrB2<sizeB2; itrB2++) {
                            for (itrB1=0; itrB1<sizeB1; itrB1++) {
                                valueB = B()({itrB1, itrB2});
                                temp1=itrA1*sizeB1+itrB1;
                                temp2=itrA2*sizeB2+itrB2;
                                C()({temp1, temp2}) = valueA*valueB;
                            }
                        }
                    }
                }
            }
            else{
                for (itrA2=0; itrA2<sizeA2; itrA2++) {
                    for (itrA1=0; itrA1<sizeA1; itrA1++) {
                        valueA = A()({itrA1, itrA2});
                        for (itrB2=0; itrB2<sizeB2; itrB2++) {
                            for (itrB1=0; itrB1<sizeB1; itrB1++) {
                                valueB = B()({itrB1, itrB2});
                                temp1=itrA1*sizeB1+itrB1;
                                temp2=itrA2*sizeB2+itrB2;
                                C()({temp1, temp2}) = alpha*valueA*valueB;
                            }
                        }
                    }
                }
            }
        }
        else{
            if (alpha==1.) {
                for (itrA2=0; itrA2<sizeA2; itrA2++) {
                    for (itrA1=0; itrA1<sizeA1; itrA1++) {
                        valueA = A()({itrA1, itrA2});
                        for (itrB2=0; itrB2<sizeB2; itrB2++) {
                            for (itrB1=0; itrB1<sizeB1; itrB1++) {
                                valueB = B()({itrB1, itrB2});
                                temp1=itrA1*sizeB1+itrB1;
                                temp2=itrA2*sizeB2+itrB2;
                                C()({temp1, temp2})=C()({temp1, temp2})*beta+valueA*valueB;
                            }
                        }
                    }
                }
            }
            else{
                for (itrA2=0; itrA2<sizeA2; itrA2++) {
                    for (itrA1=0; itrA1<sizeA1; itrA1++) {
                        valueA = A()({itrA1, itrA2});
                        for (itrB2=0; itrB2<sizeB2; itrB2++) {
                            for (itrB1=0; itrB1<sizeB1; itrB1++) {
                                valueB = B()({itrB1, itrB2});
                                temp1=itrA1*sizeB1+itrB1;
                                temp2=itrA2*sizeB2+itrB2;
                                C()({temp1, temp2})=C()({temp1, temp2})*beta + alpha*valueA*valueB;
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    template <typename T, typename major>
    inline void kron(typename gtensor<T, 2, major>::value_type alpha, const gtensor<T, 2, major> & A, const gtensor<T, 2, major> & B, typename gtensor<T, 2, major>::value_type beta, gtensor<T, 2, major> & C)
    {
        major maj;
        label::no_transpose trans;
        return tensor::detail::kron_basic(alpha, A.data(), A.size1(), A.size2(), trans, B.data(), B.size1(), B.size2(), beta, trans, C.data(), maj);
    }
    
    template<typename T, typename major>
    gtensor<T, 2, major> kron(const gtensor<T, 2, major> & A, const gtensor<T, 2, major> & B)
    {
        gtensor<T, 2, major> C(A.size1()*B.size1(), A.size2()*B.size2());
        T alpha = 1.0, beta = 0.0;
        kron(alpha, A, B, beta, C);
        return C;
    }
    
    template <typename T, typename major, typename Trans>
    inline void kron(typename gtensor<T, 2, major>::value_type alpha, const gtensor<T, 2, major> & A,
                     const tensor::matrix_trans_const_view<T, Trans, major>& B,
                     typename gtensor<T, 2, major>::value_type beta, gtensor<T, 2, major> & C)
    {
        major maj;
        typename gtensor<T, 2, major>::trans_type transA;
        typename tensor::matrix_trans_const_view<T, Trans, major>::trans_type transB;
        return tensor::detail::kron_basic(alpha, A.data(), A.size1(), A.size2(), transA, B.data(), B.size1(), B.size2(), beta, transB, C.data(), maj);
    }
    
    template<typename T, typename major, typename Trans>
    gtensor<T, 2, major> kron(const gtensor<T, 2, major> & A, const tensor::matrix_trans_const_view<T, Trans, major>& B)
    {
        gtensor<T, 2, major> C(A.size1()*B.size1(), A.size2()*B.size2());
        T alpha = 1.0, beta = 0.0;
        kron(alpha, A, B, beta, C);
        return C;
    }
    
    template <typename T, typename major, typename Trans>
    inline void kron(typename gtensor<T, 2, major>::value_type alpha, const tensor::matrix_trans_const_view<T, Trans, major>& A, const gtensor<T, 2, major> & B, typename gtensor<T, 2, major>::value_type beta, gtensor<T, 2, major> & C)
    {
        major maj;
        typename tensor::matrix_trans_const_view<T, Trans, major>::trans_type transA;
        typename gtensor<T, 2, major>::trans_type transB;
        return tensor::detail::kron_basic(alpha, A.data(), A.size1(), A.size2(), transA, B.data(), B.size1(), B.size2(), beta, transB, C.data(), maj);
    }
    
    template<typename T, typename major, typename Trans>
    gtensor<T, 2, major> kron(const tensor::matrix_trans_const_view<T, Trans, major>& A, const gtensor<T, 2, major> & B)
    {
        gtensor<T, 2, major> C(A.size1()*B.size1(), A.size2()*B.size2());
        T alpha = 1.0, beta = 0.0;
        kron(alpha, A, B, beta, C);
        return C;
    }

    
    template<typename TensorA, typename TensorB>
    inline gtensor<typename TensorA::value_type, 2, typename TensorA::orientation_type>
    kron(const tensor_expression<TensorA> & A, const tensor_expression<TensorB> & B)
    {
        gtensor<typename TensorA::value_type, 2, typename TensorA::orientation_type> C({A().size1()*B().size1(), A().size2()*B().size2()});
        typename TensorA::value_type alpha = 1.0, beta = 0.0;
        kron(alpha, A, B, beta, C);
        return C;
    }
    
    
    
    template<typename TensorA, typename TensorB, typename TensorC>
    inline gtensor<typename TensorA::value_type, 2, typename TensorA::orientation_type>
    kron3(const tensor_expression<TensorA> & A, const tensor_expression<TensorB> & B, tensor_expression<TensorC> & C)
    {
        gtensor<typename TensorA::value_type, 2, typename TensorA::orientation_type> D = kron(A, B);
        return kron(D, C);
    }
    
    
    template<typename TensorA, typename T>
    typename std::enable_if<std::is_scalar<T>::value || boost::is_complex<T>::value, gtensor<typename TensorA::value_type, 2, typename TensorA::orientation_type> >::type
    kron(const tensor_expression<TensorA> & A, T alpha)
    {
        return A*alpha;
    }
    
    template<typename TensorA, typename T>
    typename std::enable_if<std::is_scalar<T>::value || boost::is_complex<T>::value, gtensor<typename TensorA::value_type, 2, typename TensorA::orientation_type> >::type
    kron(T alpha, const tensor_expression<TensorA> & A)
    {
        return A*alpha;
    }
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value || boost::is_complex<T>::value, T >::type
    kron(T alpha, T beta)
    {
        return beta*alpha;
    }
    
    
    template<typename T, size_t N, typename major>
    typename std::enable_if<N != 2, gtensor<T, N, major> >::type
    kron(const gtensor<T, N, major> & A, const gtensor<T, N, major> & B)
    {
        typename gtensor<T, N, major>::dimvector_type dimA = A.dim(), cudimA = A.cudim(), dimB = B.dim(), dimC, index, start, end;
        size_t i,j;
        for (i = 0; i < N; ++i) { dimC[i] = dimA[i]*dimB[i];}
        gtensor<T, N, major> C(dimC);
        major maj;
        for (i = 0; i < A.size(); ++i) {
            tensor::sind2mind(i, index, cudimA, maj);
            for (j = 0; j < N; ++j) {
                start[j] = index[j]*dimB[j];
                end[j] = start[j] + dimB[j];
            }
            C(start, end) = A[i]*B;
        }
        return C;
    }
    
    template<typename T, size_t N, typename major>
    typename std::enable_if<N != 2, gtensor<T, N, major> >::type
    kron3(const gtensor<T, N, major> & A, const gtensor<T, N, major> & B, const gtensor<T, N, major> & C)
    {
        gtensor<T, N, major> D(kron(A,B));
        return kron(D,C);
    }
    
    
    
    
    template < typename Index1, typename Index2>
    std::vector<std::pair<Index1, Index2> > basis_kron(const std::vector<Index1> & A, const std::vector<Index2> & B)
    {
        std::vector<std::pair<Index1, Index2> > C;
        typename std::vector<Index1>::const_iterator itr1;
        typename std::vector<Index2>::const_iterator itr2;
        for (itr1 = A.cbegin(); itr1 != A.cend(); ++itr1) {
            for (itr2 = B.cbegin(); itr2 != B.cend(); ++itr2) {
                C.push_back(std::make_pair(*itr1, *itr2));
            }
        }
        return C;
    }
    
    template < typename size_tp1, typename size_tp2, typename size_tp3>
    std::vector<triple<size_tp1, size_tp2, size_tp3> > basis_kron3(const std::vector<size_tp1> & A, const std::vector<size_tp2> & B, const std::vector<size_tp3> & C)
    {
        std::vector<triple<size_tp1, size_tp1, size_tp3> > D;
        typename std::vector<size_tp1>::const_iterator itr1;
        typename std::vector<size_tp2>::const_iterator itr2;
        typename std::vector<size_tp3>::const_iterator itr3;
        for (itr1 = A.cbegin(); itr1 != A.cend(); ++itr1) {
            for (itr2 = B.cbegin(); itr2 != B.cend(); ++itr2) {
                for (itr3 = C.cbegin(); itr3 != C.cend(); ++itr3) {
                    D.push_back(make_triple(*itr1, *itr2, *itr3));
                }
            }
        }
        return D;
    }

    
}




#endif
