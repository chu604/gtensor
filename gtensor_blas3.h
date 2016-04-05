

#ifndef guochu_gtensor_gtensor_blas3_h
#define guochu_gtensor_gtensor_blas3_h

#include "gtensor.h"
#include <boost/numeric/bindings/blas/level3/gemm.hpp>


//C = A*B, case distinguished by (1) A.rank =2 && B.rank =2, (2) A.rank =2 && B.rank != 2, (3) A.rank !=2 && B.rank = 2, (4) A.rank !=2 && B.rank != 2

namespace guochu {
    
 
    // case A and B are both not of dimension 2
    template <typename T, size_t N, size_t N1>
    typename std::enable_if<(N!=2) && (N1!=2), void>::type
    gemm(T alpha, const gtensor<T, N> & A, const gtensor<T, N1> & B, T beta, gtensor<T, N+N1-2> & C)
    {
        int size1 = 1, size2, size3 = 1, i;
        typename gtensor<T, N>::dimvector_type      Adim = A.dim();
        typename gtensor<T, N1>::dimvector_type     Bdim = B.dim();
        for (i = 0; i < N-1; ++i) {size1*=Adim[i];}
        for (i = 0; i < N1-1; ++i) {size3*=Bdim[i+1];}
        size2 = (int)Adim[N-1];
#ifdef GTENSOR_DEBUG
        if (A.size() == 0) {throw std::runtime_error("gemm(A,B,C), the size of A is 0.\n");}
        if (B.size() == 0) {throw std::runtime_error("gemm(A,B,C), the size of B is 0.\n");}
        typename gtensor<T, N+N1-2>::dimvector_type Cdim = C.dim();
        for (size_t i = 0; i < N-1; ++i) {
            if (Adim[i] != Cdim[i]) {
                throw std::runtime_error("gemm(A,B,C), the dimension of A and C does not match.\n");
            }
        }
        if (Adim[N-1] != Bdim[0]) {
            throw std::runtime_error("gemm(A,B,C), the dimension of A and B does not match.\n");
        }
        for (size_t i = 1; i < N1; ++i) {
            if (Bdim[i] != Cdim[N-2+i]) {
                throw std::runtime_error("gemm(A,B,C), the dimension of B and C does not match.\n");
            }
        }
#endif
        boost::numeric::bindings::blas::detail::gemm('N', 'N', size1, size3,
                                                     size2, alpha, A.data(), size1, B.data(), size2, beta, C.data(), size1);
    }
    
//    template <typename T, size_t N, size_t N1>
//    inline typename std::enable_if<(N!=2) && (N1!=2), void>::type
//    gemm(const gtensor<T, N> & A, const gtensor<T, N1> & B, gtensor<T, N+N1-2> & C)
//    {
//        T alpha = 1.0, beta = 0.0;
//        return gemm(alpha, A, B, beta, C);
//    }
    
    // case A is dimension 2 but B is not
    template <typename MatA, size_t N>
    typename std::enable_if<std::tuple_size<typename MatA::dimvector_type>::value==2 && (N!=2), void>::type
    gemm(typename MatA::value_type alpha, const tensor_expression<MatA> & A, const gtensor<typename MatA::value_type, N> & B, typename MatA::value_type beta, gtensor<typename MatA::value_type, N> & C)
    {
        std::array<size_t, N> Bdim = B.dim();
        int size3 = 1;
        for (size_t i = 0; i < N-1; ++i) {size3*=Bdim[i+1];}
#ifdef GTENSOR_DEBUG
        assert(A().rank() == 2 && B.rank() == C.rank());
        std::array<size_t, N> Cdim = C.dim();
        if (A().size1() != C.size1()) {
            throw std::runtime_error("gemm(2,N,N)(A,B,C), the dimension of A and C does not match.\n");
        }
        int sizeC3 = 1;
        for (size_t i = 0; i < N-1; ++i) {sizeC3*=Cdim[i+1];}
        if (A().size2() != B.size1()) {
            throw std::runtime_error("gemm(2,N,N)(A,B,C), the dimension of A and B does not match.\n");
        }
        if (sizeC3 != size3) {
            throw std::runtime_error("gemm(2,N,N)(A,B,C), the dimension of B and C does not match.\n");
        }
#endif
        label::column_major maj;
        typename MatA::trans_type tra;
        label::no_transpose trb;
        boost::numeric::bindings::blas::detail::gemm(maj,
                                     tra, trb, (int)A().size1(), size3,
                                     (int)A().size2(), alpha, A().data(), (int)A().leadingdim(), B.data(), (int)A().size2(),
                                     beta, C.data(), (int)A().size1());
    }
    
//    template <typename MatA, size_t N>
//    inline void gemm(const tensor_expression<MatA> & A, const gtensor<typename MatA::value_type, N> & B, gtensor<typename MatA::value_type, N> & C)
//    {
//        typename MatA::value_type alpha = 1.0, beta = 0.0;
//        return gemm(alpha, A, B, beta, C);
//    }
    
    
    // case B is dimension 2 but A is not
    template <typename MatB, size_t N>
    typename std::enable_if< (N!=2) && std::tuple_size<typename MatB::dimvector_type>::value==2, void>::type
    gemm(typename MatB::value_type alpha, const gtensor<typename MatB::value_type, N> & A, const tensor_expression<MatB> & B, typename MatB::value_type beta, gtensor<typename MatB::value_type, N> & C)
    {
        std::array<size_t, N> Adim = A.dim();
        int size1 = 1;
        for (size_t i = 0; i < N-1; ++i) {size1*=Adim[i];}
#ifdef GTENSOR_DEBUG
        assert(A.rank() == C.rank() && B().rank() == 2);
        std::array<size_t, N> Cdim = C.dim();
        if (A.size1() != C.size1()) {
            throw std::runtime_error("gemm(N,2,N)(A,B,C), the dimension of A and C does not match.\n");
        }
        int sizeC1 = 1;
        for (size_t i = 0; i < N-1; ++i) {sizeC1*=Cdim[i];}
        if (Adim[N-1] != B().size1()) {
            throw std::runtime_error("gemm(N,2,N)(A,B,C), the dimension of A and B does not match.\n");
        }
        if (sizeC1 != size1) {
            throw std::runtime_error("gemm(N,2,N)(A,B,C), the dimension of B and C does not match.\n");
        }
#endif
        label::column_major maj;
        label::no_transpose tra;
        typename MatB::trans_type trb;
        boost::numeric::bindings::blas::detail::gemm(maj,
                                     tra, trb, size1, (int)B().size2(),
                                     (int)B().size1(), alpha, A.data(), size1, B().data(), (int)B().leadingdim(),
                                     beta, C.data(), size1);
    }
    
    
//    template <typename MatB, size_t N>
//    inline void gemm(const gtensor<typename MatB::value_type, N> & A, const tensor_expression<MatB> & B, gtensor<typename MatB::value_type, N> & C)
//    {
//        typename MatB::value_type alpha = 1.0, beta = 0.0;
//        return gemm(alpha, A, B, beta, C);
//    }
    
    
    // case A and B are both not of dimension 2
    template <typename T, size_t N, size_t N1>
    typename std::enable_if<(N!=2) && (N1!=2), gtensor<T, N+N1-2> >::type
    operator*(const gtensor<T, N> & A, const gtensor<T, N1> & B)
    {
        int size1 = 1, size2, size3 = 1, i;
        typename gtensor<T, N>::dimvector_type      Adim = A.dim();
        typename gtensor<T, N1>::dimvector_type     Bdim = B.dim();
        typename gtensor<T, N+N1-2>::dimvector_type  Cdim;
        for (i = 0; i < N-1; ++i) {
            Cdim[i] = Adim[i];
            size1*=Adim[i];
        }
        for (i = 0; i < N1-1; ++i) {
            Cdim[i+N-1] = Bdim[i+1];
            size3*=Bdim[i+1];
        }
        gtensor<T, N+N1-2> C(Cdim);
        size2 = (int)Adim[N-1];
#ifdef GTENSOR_DEBUG
        if (A.size() == 0) {throw std::runtime_error("C = A*B, the size of A is 0.\n");}
        if (B.size() == 0) {throw std::runtime_error("C = A*B, the size of B is 0.\n");}
        if (Adim[N-1] != Bdim[0]) {
            throw std::runtime_error("C = A*B, the dimension of A and B does not match.\n");
        }
#endif
        T alpha = 1., beta = 0.;
        boost::numeric::bindings::blas::detail::gemm('N', 'N', size1, size3,
                                                     size2, alpha, A.data(), size1, B.data(), size2, beta, C.data(), size1);
        return C;
    }
    
    // case A is dimension 2 but B is not
    template <typename MatA, size_t N>
    typename std::enable_if<std::tuple_size<typename MatA::dimvector_type>::value==2&&(N != 2), gtensor<typename MatA::value_type, N> >::type
    operator*(const tensor_expression<MatA> & A, const gtensor<typename MatA::value_type, N> & B)
    {
        std::array<size_t, N> Bdim = B.dim(), Cdim;
        int size3 = 1;
        for (size_t i = 1; i < N; ++i) {
            Cdim[i] = Bdim[i];
            size3*=Bdim[i];
        }
        Cdim[0] = A().size1();
        gtensor<typename MatA::value_type, N> C(Cdim);
#ifdef GTENSOR_DEBUG
        if (A().size() == 0) {throw std::runtime_error("C = A*B, the size of A is 0.\n");}
        if (B.size() == 0) {throw std::runtime_error("C = A*B, the size of B is 0.\n");}
        if (A().size2() != Bdim[0]) {
            throw std::runtime_error("C = A*B, the dimension of A and B does not match.\n");
        }
#endif
        typename MatA::value_type alpha = 1., beta = 0.;
        label::column_major maj;
        typename MatA::trans_type tra;
        label:: no_transpose trb;
       boost::numeric::bindings::blas::detail::gemm(maj,
                                     tra, trb, (int)A().size1(), size3,
                                     (int)A().size2(), alpha, A().data(), (int)A().leadingdim(), B.data(), (int)A().size2(),
                                     beta, C.data(), (int)A().size1());
        return C;
    }
    
    
    // case B is dimension 2 but A is not
    template <typename MatB, size_t N>
    typename std::enable_if<(N != 2) && std::tuple_size<typename MatB::dimvector_type>::value==2, gtensor<typename MatB::value_type, N> >::type
    operator*(const gtensor<typename MatB::value_type, N> & A, const tensor_expression<MatB> & B)
    {
        std::array<size_t, N> Adim = A.dim(), Cdim;
        int size1 = 1;
        for (size_t i = 0; i < N-1; ++i) {
            Cdim[i] = Adim[i];
            size1*=Adim[i];
        }
        Cdim[N-1] = B().size2();
        gtensor<typename MatB::value_type, N> C(Cdim);
#ifdef GTENSOR_DEBUG
        if (A.size() == 0) {throw std::runtime_error("C = A*B, the size of A is 0.\n");}
        if (B().size() == 0) {throw std::runtime_error("C = A*B, the size of B is 0.\n");}
        if (Adim[N-1] != B().size1()) {
            throw std::runtime_error("C = A*B, the dimension of A and B does not match.\n");
        }
#endif
        typename MatB::value_type alpha = 1., beta = 0.;
         label::column_major maj;
        label::no_transpose tra;
        typename MatB::trans_type trb;
        boost::numeric::bindings::blas::detail::gemm(maj,
                                     tra, trb, size1, (int)B().size2(),
                                     (int)B().size1(), alpha, A.data(), size1, B().data(), (int)B().leadingdim(),
                                     beta, C.data(), size1);
        return C;
    }
    
    
    
    /*************in case we know that one of the tensor is identity*************/
    template <typename T, size_t N>
    const gtensor<T, N> &
    operator*(const gtensor<T, N> & A, label::identity B)
    {
        return A;
    }
    
    template <typename T, size_t N>
    const gtensor<T, N> &
    operator*(label::identity B, const gtensor<T, N> & A)
    {
        return A;
    }
    /*************in case we know that one of the tensor is identity*************/
    
    
    template <typename MatA, typename MatB, typename MatC>
    typename std::enable_if<std::tuple_size<typename MatA::dimvector_type>::value==2 &&std::tuple_size<typename MatB::dimvector_type>::value==2, void>::type
    gemm(typename MatA::value_type alpha, const tensor_expression<MatA> & A, const tensor_expression<MatB> & B, typename MatA::value_type beta, tensor_expression<MatC> & C)
    {
#ifdef GTENSOR_DEBUG
        assert(A().rank() == 2 && B().rank() ==2 && C().rank()==2);
        if (A().size1() != C().size1()) {
            throw std::runtime_error("gemm(2,2,2)(A,B,C), the dimension of A and C does not match.\n");
        }
        if (A().size2() != B().size1()) {
            throw std::runtime_error("gemm(2,2,2)(A,B,C), the dimension of A and B does not match.\n");
        }
        if (C().size2() != B().size2()) {
            throw std::runtime_error("gemm(2,2,2)(A,B,C), the dimension of B and C does not match.\n");
        }
#endif
         label::column_major maj;
        typename MatA::trans_type tra;
        typename MatB::trans_type trb;
        boost::numeric::bindings::blas::detail::gemm(maj,
                                     tra, trb, (int)A().size1(), (int)B().size2(),
                                     (int)A().size2(), alpha, A().data(), (int)A().leadingdim(), B().data(), (int)B().leadingdim(),
                                     beta, C().data(), (int)C().leadingdim());
    }
    
    
//    template <typename MatA, typename MatB, typename MatC>
//    inline void gemm(const tensor_expression<MatA> & A, const tensor_expression<MatB> & B, tensor_expression<MatC> & C)
//    {
//        typename MatA::value_type alpha = 1.0, beta = 0.0;
//        return gemm(alpha, A, B, beta, C);
//    }
    
    
    
    template <typename MatA, typename MatB>
    typename std::enable_if<std::tuple_size<typename MatA::dimvector_type>::value==2 &&std::tuple_size<typename MatB::dimvector_type>::value==2, gtensor<typename MatA::value_type, 2> >::type
    operator*(const tensor_expression<MatA> & A, const tensor_expression<MatB> & B)
    {
#ifdef GTENSOR_DEBUG
        assert(A().rank() == 2 && B().rank() ==2);
        if (A().size2() != B().size1()) {
            throw std::runtime_error("C=A*B(2,2,2), the dimension of A and B does not match.\n");
        }
#endif
        typename MatA::value_type alpha = 1.0, beta = 0.0;
        gtensor<typename MatA::value_type, 2> C({A().size1(), B().size2()});
        // assert(A().leadingdim() == A().size1() && B().leadingdim() == B().size1() && C.leadingdim() == C.size1());
         label::column_major maj;
        typename MatA::trans_type tra;
        typename MatB::trans_type trb;
        boost::numeric::bindings::blas::detail::gemm(maj,
                                     tra, trb, (int)A().size1(), (int)B().size2(),
                                     (int)A().size2(), alpha, A().data(), (int)A().leadingdim(), B().data(), (int)B().leadingdim(),
                                     beta, C.data(), (int)C.leadingdim());
        return C;
    }
    
    
    
    
    template<size_t N2, size_t N4, typename T, size_t N1, size_t N3>
    gtensor<T, N1-N2+N3-N4>
    contracttensors(const gtensor<T, N1> & A, std::array<size_t, N2> indexA, const gtensor<T, N3> & B, std::array<size_t, N4> indexB)
    {
#ifdef GTENSOR_DEBUG
        if (N1+N3 == N2+N4) {
            std::cout << "contracttensors(A(N1), indexA(N2), B(N3), indexB(N4)), N1+N3 == N2+N4, the results will be a scalar(tensor<0>)"<<std::endl;
        }
#endif
        std::array<size_t, N1> indexApermute, Adim = A.dim();
        std::array<size_t, N3> indexBpermute, Bdim = B.dim();
        std::array<size_t, N1-N2+N3-N4> dimC;
        int sizeindexA=1, sizeindexB=1, sizeindexArest = 1, sizeindexBrest = 1, i;
        for (i=0; i < N2; ++i) {sizeindexA*=Adim[indexA[i]-1];}
        for (i=0; i < N4; ++i) {sizeindexB*=Bdim[indexB[i]-1];}
        assert(sizeindexA == sizeindexB);
        for (i = N1-N2; i < N1; ++i) {indexApermute[i] = indexA[i-N1+N2];}
        for (i = 0; i < N4; ++i){indexBpermute[i] = indexB[i];}
        std::vector<size_t> indexArest(N1), indexBrest(N3);
        for (i = 0; i < N1; ++i) { indexArest[i] = i+1;}
        for (i = 0; i < N3; ++i) { indexBrest[i] = i+1;}
        std::sort(indexA.begin(), indexA.end());
        std::sort(indexB.begin(), indexB.end());
        for (i=N2-1; i>=0; --i) {indexArest.erase(indexArest.begin()+indexA[i]-1);}
        for (i=N4-1; i>=0; --i) {indexBrest.erase(indexBrest.begin()+indexB[i]-1);}
        for (i=0; i<N1-N2; ++i) {sizeindexArest*=Adim[indexArest[i]-1]; dimC[i] = Adim[indexArest[i]-1];}
        for (i=0; i<N3-N4; ++i) {sizeindexBrest*=Bdim[indexBrest[i]-1]; dimC[i+N1-N2] = Bdim[indexBrest[i]-1];}
        for (i = 0; i <  N1-N2; ++i) {indexApermute[i] = indexArest[i];}
        for (i = N4; i < N3; ++i){indexBpermute[i] = indexBrest[i-N4];}
        gtensor<T, N1> Atemp;
        gtensor<T, N3> Btemp;
        permute(A, Atemp, indexApermute);
        permute(B, Btemp, indexBpermute);
        gtensor<T, N1-N2+N3-N4> C(dimC);
        T alpha = 1.0, beta = 0.0;
        boost::numeric::bindings::blas::detail::gemm('N', 'N', sizeindexArest, sizeindexBrest, sizeindexA, alpha, Atemp.data(), sizeindexArest, Btemp.data(), sizeindexB, beta, C.data(), sizeindexArest);
        return C;
    }
    
    
    /*************in case we know that one of the tensor is identity*************/
    template<typename T, size_t N1>
    const gtensor<T, N1> &
    contracttensors(const gtensor<T, N1> & A, std::array<size_t, 2> indexA, label::identity B, std::array<size_t, 2> indexB)
    {
        return A;
    }
    template<typename T, size_t N1>
    const gtensor<T, N1> &
    contracttensors(label::identity B, std::array<size_t, 2> indexB, const gtensor<T, N1> & A, std::array<size_t, 2> indexA)
    {
        return A;
    }
    /*************in case we know that one of the tensor is identity*************/
    

    
}


#endif