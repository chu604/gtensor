

#ifndef guochu_gtensor_tensorop_h
#define guochu_gtensor_tensorop_h

#include "gtensor.h"
#include <boost/numeric/bindings/blas/level3/gemm.hpp>

namespace guochu {
    /**********************************************************************
     a two dimwnsional tensor, which is assumed to be a physical operation,
     acts on the middle index on three dimensional and four dimensional
     tensor.
     ********************************************************************/
    
    template<typename T>
    void prod32_21(T alpha, const gtensor<T, 3> & A, const gtensor<T, 2> & B, T beta, gtensor<T, 3> & C)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D2new = (int)B.size2(), i;
        //T alpha = 1.0, beta = 0.0;
        for (i = 0; i < D3; ++i) {
            boost::numeric::bindings::blas::detail::gemm('N', 'N', D1, D2new, D2, alpha, &A(0,0,i), D1, &B(0,0), D2, beta, &C(0,0,i), D1);
        }
    }
    
    template<typename T>
    gtensor<T, 3> prod32_21(const gtensor<T, 3> & A, const gtensor<T, 2> & B)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D2new = (int)B.size2();
        assert(D2 == B.size1());
        gtensor<T, 3> C(D1, D2new, D3);
        T alpha = 1.0, beta = 0.0;
        prod32_21(alpha, A, B, beta, C);
        return C;
    }
    
    template<typename T>
    void prod42_21(T alpha, const gtensor<T, 4> & A, const gtensor<T, 2> & B, T beta, gtensor<T, 4> & C)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D4 = (int)A.size4(), D2new = (int)B.size2(), i, j;
        //T alpha = 1.0, beta = 0.0;
#ifdef GTENSOR_DEBUG
        assert(A.size1() == C.size1()&&A.size3()==C.size3()&&A.size4()==C.size4());
        assert(B.size2() == C.size2());
#endif
        for (i = 0; i < D4; ++i) {
            for (j = 0; j < D3; ++j) {
                boost::numeric::bindings::blas::detail::gemm('N', 'N', D1, D2new, D2, alpha, &A(0,0,j,i), D1, &B(0,0), D2, beta, &C(0,0,j,i), D1);
            }
        }
    }
    
    
    template<typename T>
    gtensor<T, 4> prod42_21(const gtensor<T, 4> & A, const gtensor<T, 2> & B)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D4 = (int)A.size4(), D2new = (int)B.size2();
        assert(D2 == B.size1());
        gtensor<T, 4> C(D1, D2new, D3, D4);
        T alpha = 1.0, beta = 0.0;
        prod42_21(alpha, A, B, beta, C);
        return C;
    }

    
    
    template<typename T>
    void prod43_21(T alpha, const gtensor<T, 4> & A, const gtensor<T, 2> & B, T beta, gtensor<T, 4> & C)
    {
#ifdef GTENSOR_DEBUG
        assert(A.size1()==C.size1()&&A.size2()==C.size2()&&A.size4()==C.size4());
        assert(B.size2()==C.size3());
#endif
        int D1 = int(A.size1()*A.size2()), D2 = (int)A.size3(), D3 = (int)A.size4(), D2new = (int)B.size2(), i;
        //T alpha = 1.0, beta = 0.0;
        for (i = 0; i < D3; ++i) {
            boost::numeric::bindings::blas::detail::gemm('N', 'N', D1, D2new, D2, alpha, &A(0,0,0,i), D1, &B(0,0), D2, beta, &C(0,0,0,i), D1);
        }
    }
    
    template<typename T>
    gtensor<T, 4> prod43_21(const gtensor<T, 4> & A, const gtensor<T, 2> & B)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D4 = (int)A.size4(), D3new = (int)B.size2();
        assert(D3 == B.size1());
        gtensor<T, 4> C(D1, D2, D3new, D4);
        T alpha = 1.0, beta = 0.0;
        prod43_21(alpha, A, B, beta, C);
        return C;
    }


    
    template<typename T>
    void prod22_32(T alpha, const gtensor<T, 2> & B, const gtensor<T, 3> & A, T beta, gtensor<T, 3> & C)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D2new = (int)B.size1(), i;
        //assert(A.size1()==C.size1()&&A.size3()==C.size3()&&A.size2()==B.size2()&&C.size2()==B.size1());
        //T alpha = 1.0, beta = 0.0;
        for (i = 0; i < D3; ++i) {
            boost::numeric::bindings::blas::detail::gemm('N', 'T', D1, D2new, D2, alpha, &A(0,0,i), D1, &B(0,0), D2new, beta, &C(0,0,i), D1);
        }
    }
    
    
    template<typename T>
    gtensor<T, 3> prod22_32(const gtensor<T, 2> & B, const gtensor<T, 3> & A)
    {
        int D1 = (int)A.size1(), D2 = (int)A.size2(), D3 = (int)A.size3(), D2new = (int)B.size1();
        assert(D2 == B.size2());
        gtensor<T, 3> C(D1, D2new, D3);
        T alpha = 1.0, beta = 0.0;
        prod22_32(alpha, B, A, beta, C);
        return C;
    }

    
    template<typename T>
    void prod22_42(T alpha, const gtensor<T, 2> & B, const gtensor<T, 4> & A, T beta, gtensor<T, 4> & C)
    {
        int D1 = A.size1(), D2 = A.size2(), D3 = A.size3(), D4 = A.size4(), D2new = B.size1(), i, j;
#ifdef GTENSOR_DEBUG
        assert(A.size1()==C.size1()&&A.size3()==C.size3()&&A.size2()==B.size2()&&C.size2()==B.size1()&&A.size4()==C.size4());
#endif
        //T alpha = 1.0, beta = 0.0;
        for (i = 0; i < D4; ++i) {
            for (j = 0; j < D3; ++j) {
                boost::numeric::bindings::blas::detail::gemm('N', 'T', D1, D2new, D2, alpha, &A(0,0,j,i), D1, &B(0,0), D2new, beta, &C(0,0,j,i), D1);
            }
        }
    }
    
    template<typename T>
    gtensor<T, 4> prod22_42(const gtensor<T, 2> & B, const gtensor<T, 4> & A)
    {
        int D1 = A.size1(), D2 = A.size2(), D3 = A.size3(), D4 = A.size4(), D2new = B.size1();
        assert(D2 == B.size2());
        gtensor<T, 4> C(D1, D2new, D3, D4);
        T alpha = 1.0, beta = 0.0;
        prod22_42(alpha, B, A, beta, C);
        return C;
    }

    
    template<typename T>
    void prod22_43(T alpha, const gtensor<T, 2> & B, const gtensor<T, 4> & A, T beta, gtensor<T, 4> & C)
    {
        int D1 = A.size1()*A.size2(), D2 = A.size3(), D3 = A.size4(), D2new = B.size1(), i;
#ifdef GTENSOR_DEBUG
        assert(A.size1()==C.size1()&&A.size2()==C.size2()&&A.size4()==C.size4()&&A.size3()==B.size2()&&C.size3()==B.size1());
#endif
        //T alpha = 1.0, beta = 0.0;
        for (i = 0; i < D3; ++i) {
            boost::numeric::bindings::blas::detail::gemm('N', 'T', D1, D2new, D2, alpha, &A(0,0,0,i), D1, &B(0,0), D2new, beta, &C(0,0,0,i), D1);
        }
    }
    
    template<typename T>
    gtensor<T, 4> prod22_43(const gtensor<T, 2> & B, const gtensor<T, 4> & A)
    {
        int D1 = A.size1(), D2 = A.size2(), D3 = A.size3(), D4 = A.size4(), D3new = B.size1();
        assert(D3 == B.size2());
        gtensor<T, 4> C(D1, D2, D3new, D4);
        T alpha = 1.0, beta = 0.0;
        prod22_43(alpha, B, A, beta, C);
        return C;
    }


    
    template<typename T>
    void prod312_312(T alpha, const gtensor<T, 3> & A, const gtensor<T, 3> & B, T beta, gtensor<T, 2> & C)
    {
#ifdef GTENSOR_DEBUG
        assert(!A.empty() && !B.empty() && !C.empty());
        assert(A.size1()*A.size2() ==  B.size1()*B.size2());
        assert(A.size3() == C.size1() && B.size3() == C.size2());
#endif
        int sizeA1 = (int)A.size1(), sizeA2 = (int)A.size2(), sizeA3 = (int)A.size3(), sizeB3 = (int)B.size3(), size1 = sizeA1*sizeA2;
        boost::numeric::bindings::blas::detail::gemm('T', 'N', sizeA3, sizeB3, size1, alpha, A.data(), size1, B.data(), size1, beta, C.data(), sizeA3);
    }
    
    template<typename T>
    gtensor<T, 2> prod312_312(const gtensor<T, 3> & A, const gtensor<T, 3> & B)
    {
        gtensor<T, 2> C(A.size3(), B.size3());
        T alpha = 1.0, beta = 0.0;
        prod312_312(alpha, A, B, beta, C);
        return C;
    }
    
    template<typename T>
    void prod323_323(T alpha, const gtensor<T, 3> & A, const gtensor<T, 3> & B, T beta, gtensor<T, 2> & C)
    {
#ifdef GTENSOR_DEBUG
        assert(!A.empty() && !B.empty() && !C.empty());
        assert(A.size2()*A.size3() ==  B.size2()*B.size3());
        assert(A.size1() == C.size1() && B.size1() == C.size2());
#endif
        int sizeA1 = (int)A.size1(), sizeA2 = (int)A.size2(), sizeA3 = (int)A.size3(), sizeB1 = (int)B.size1(), size2 = sizeA2*sizeA3;
        boost::numeric::bindings::blas::detail::gemm('N', 'T', sizeA1, sizeB1, size2, alpha, A.data(), sizeA1, B.data(), sizeB1, beta, C.data(), sizeA1);
    }
    
    template<typename T>
    gtensor<T, 2> prod323_323(const gtensor<T, 3> & A, const gtensor<T, 3> & B)
    {
        gtensor<T, 2> C(A.size1(), B.size1());
        T alpha = 1.0, beta = 0.0;
        prod323_323(alpha, A, B, beta, C);
        return C;
    }
    
    
    template<typename T>
    void prod31_33(T alpha, const gtensor<T, 3> & A, const gtensor<T, 3> & B, T beta, gtensor<T, 4> & C)
    {
#ifdef GTENSOR_DEBUG
        assert(!A.empty() && !B.empty() && !C.empty());
        assert(A.size1() ==  B.size3());
        assert(A.size2() ==  C.size1() && A.size3() == C.size2() && B.size1() == C.size3() && B.size2() == C.size4());
#endif
        int sizeA2 = A.size2(), sizeA3 = A.size3(), sizeB1 = B.size1(), sizeB2 = B.size2(), sizem = A.size1(), sizeA23 = sizeA2*sizeA3, sizeB12 = sizeB1*sizeB2;
        boost::numeric::bindings::blas::detail::gemm('T', 'T', sizeA23, sizeB12, sizem, alpha, A.data(), sizem, B.data(), sizeB12, beta, C.data(), sizeA23);
    }
    
    template<typename T>
    gtensor<T, 4> prod31_33(const gtensor<T, 3> & A, const gtensor<T, 3> & B)
    {
        T alpha = 1.0, beta = 0.0;
        std::array<size_t, 4> Cindex;
        Cindex[0] = A.size2(), Cindex[1] = A.size3(), Cindex[2] = B.size1(), Cindex[3] = B.size2();
        gtensor<T, 4> C(Cindex);
        prod31_33(alpha, A, B, beta, C);
        return C;
    }
    
    
    template<typename T>
    void prod434_412(T alpha, const gtensor<T, 4> & A, const gtensor<T, 4> & B, T beta, gtensor<T, 4> & C)
    {
#ifdef GTENSOR_DEBUG
        assert(!A.empty() && !B.empty() && !C.empty());
        assert(A.size3() ==  B.size1() && A.size4() == B.size2());
        assert(A.size1() ==  C.size1() && A.size2() == C.size2() && B.size3() == C.size3() && B.size4() == C.size4());
#endif
        int size1 = A.size1()*A.size2(), size2 = A.size3()*A.size4(), size3 = B.size3*B.size4();
        boost::numeric::bindings::blas::detail::gemm('N', 'N', size1, size3, size2, alpha, A.data(), size1, B.data(), size2, beta, C.data(), size1);
    }
    
    template<typename T>
    gtensor<T, 4> prod434_412(const gtensor<T, 4> & A, const gtensor<T, 4> & B)
    {
        gtensor<T, 4> C(A.size1(), A.size2(), B.size3(), B.size4());
        T alpha = 1.0, beta = 0.0;
        prod434_412(alpha, A, B, beta, C);
        return C;
    }


    
    
    
    

    

}


#endif



