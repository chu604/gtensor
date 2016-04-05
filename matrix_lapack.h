
#ifndef guochu_gtensor_matrix_lapack_h
#define guochu_gtensor_matrix_lapack_h


#include "gtensor.h"
#include <boost/numeric/bindings/lapack/driver/gesdd.hpp>
#include <boost/numeric/bindings/lapack/driver/heev.hpp>
#include <boost/numeric/bindings/lapack/driver/heevx.hpp>
#include <boost/numeric/bindings/lapack/driver/gesv.hpp>
#include <boost/numeric/bindings/lapack/driver/geev.hpp>
#include <boost/numeric/bindings/blas/level1/iamax.hpp>
#include <boost/numeric/bindings/uplo_tag.hpp>


namespace guochu {
    
    //general linear solver of aX = b, on output, a stores the LU decomposition, and b stores X.
    template<typename T>
    inline size_t
    gesv(gtensor<T, 2> & a, gtensor<T, 2> & b, std::vector<int> & ipiv)
    {
#ifdef GTENSOR_DEBUG
        if (a.size1() != a.size2() || a.size2() != b.size1() || ipiv.size() < a.size2()) {
            throw std::runtime_error("a/b, the size of a and b does not match.\n");
        }
#endif
        return boost::numeric::bindings::lapack::detail::gesv(label::column_major(), (int)a.size2(), (int)b.size2(), a.data(), (int)a.size1(), ipiv.data(), b.data(), (int)b.size1());
    }
    
    template<typename T>
    inline const gtensor<T, 2> &
    operator/(gtensor<T, 2> & b, gtensor<T, 2> & a)
    {
        std::vector<int> ipiv(a.size1());
        size_t info = gesv(a, b, ipiv);
        if (info != 0) {
            throw std::runtime_error("gesv(a, b)(b/a) fails.\n");
        }
        return b;
    }
    
    
    
    
 
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, size_t>::type
    gesdd(gtensor<T, 2> & A, diagmatrix<T> & S, gtensor<T, 2> & U, gtensor<T, 2> & V)
    {
        int size1 = (int)A.size1(), size2 = (int)A.size2(), minisize = std::min(size1, size2);
        if (A.empty()) { throw std::runtime_error("[U,S,V] = svd(A), A is empty.\n");}
        S.resize(minisize);
        U.resize({A.size1(), size_t(minisize)});
        V.resize({size_t(minisize), A.size2()});
        T wkopt;
        int lwork = -1;
        size_t info;
        std::vector<int> iwork(8*size1);
        info = boost::numeric::bindings::lapack::detail::gesdd('S', size1, size2, A.data(), size1, S.data(), U.data(), size1, V.data(), minisize, &wkopt, lwork, iwork.data());
        lwork = (int)wkopt;
        std::vector<T> work(lwork);
        boost::numeric::bindings::lapack::detail::gesdd('S', size1, size2, A.data(), size1, S.data(), U.data(), size1, V.data(), minisize, work.data(), lwork, iwork.data());
        return info;
    }
    
    template<typename T>
    typename std::enable_if<boost::is_complex<T>::value, size_t>::type
    gesdd(gtensor<T, 2> & A, diagmatrix<typename boost::numeric::bindings::traits::type_traits<T>::real_type> & S, gtensor<T, 2> & U, gtensor<T, 2> & V)
    {
        if (A.empty()) { throw std::runtime_error("A = U*S*V, A is empty.\n");}
        int size1 = (int)A.size1(), size2 = (int)A.size2(), minisize = std::min(size1, size2);
        S.resize(minisize);
        U.resize({A.size1(), size_t(minisize)});
        V.resize({size_t(minisize), A.size2()});
        std::vector<typename boost::numeric::bindings::traits::type_traits<T>::real_type> rwork(5*size1*size1 + 7*size1); //this cause me sutle bug if the length of rwork is not long enough, which is really hard to debug!!!
        T wkopt;
        int lwork = -1;
        size_t info;
        std::vector<int> iwork(8*size1);
        info = boost::numeric::bindings::lapack::detail::gesdd('S', size1, size2, A.data(), size1, S.data(), U.data(), size1, V.data(), minisize, &wkopt, lwork, rwork.data(), iwork.data());
        lwork = (int)std::real(wkopt);
        std::vector<T> work(lwork);
        info = boost::numeric::bindings::lapack::detail::gesdd('S', size1, size2, A.data(), size1, S.data(), U.data(), size1, V.data(), minisize, work.data(), lwork, rwork.data(), iwork.data());
        return info;
    }
    
    template<typename T>
    typename std::enable_if<boost::is_scalar<T>::value, size_t>::type
    heev(gtensor<T, 2> & A, diagmatrix<T> & S)
    {
        if (A.empty()) { throw std::runtime_error("eig(A), A is empty.\n");}
        int size1 = (int)A.size1(), size2 = (int)A.size2(), lwork = -1;
        size_t info;
        if (size1 != size2) { throw std::runtime_error("eig(A), A must be a square matrix.\n");}
        S.resize(size1);
        std::vector<T> work;
        T wkopt;
        info = boost::numeric::bindings::lapack::detail::heev('V', boost::numeric::bindings::tag::lower(), size1, A.data(), size1, S.data(), &wkopt, lwork);
        lwork = (int)wkopt;
        work.resize(lwork);
        info = boost::numeric::bindings::lapack::detail::heev('V', boost::numeric::bindings::tag::lower(), size1, A.data(), size1, S.data(), work.data(), lwork);
        return info;
    }
    
    template<typename T>
    typename std::enable_if<boost::is_complex<T>::value, size_t>::type
    heev(gtensor<T, 2> & A, diagmatrix<typename boost::numeric::bindings::traits::type_traits<T>::real_type> & S)
    {
        if (A.empty()) { throw std::runtime_error("eig(A), A is empty.\n");}
        int size1 = (int)A.size1(), size2 = (int)A.size2(), lwork = -1;
        size_t info;
        if (size1 != size2) { throw std::runtime_error("eig(A), A must be a square matrix.\n");}
        S.resize(size1);
        std::vector<typename boost::numeric::bindings::traits::type_traits<T>::real_type> rwork(3*size1);
        std::vector<T> work;
        T wkopt;
        info = boost::numeric::bindings::lapack::detail::heev('V', boost::numeric::bindings::tag::lower(), size1, A.data(), size1, S.data(), &wkopt, lwork, rwork.data());
        lwork = (int)std::real(wkopt);
        work.resize(lwork);
        info = boost::numeric::bindings::lapack::detail::heev('V', boost::numeric::bindings::tag::lower(), size1, A.data(), size1, S.data(), work.data(), lwork, rwork.data());
        return info;
    }
    
    
    template<typename T>
    typename std::enable_if<boost::is_scalar<T>::value, size_t>::type
    heevx(gtensor<T, 2> & A, diagmatrix<T> & S, gtensor<T, 2> & V, int il, int iu,
     typename boost::numeric::bindings::traits::type_traits<T>::real_type abstol = 1.0e-12)
    {
        if (A.empty()) { throw std::runtime_error("heevx(A), A is empty.\n");}
        int size1 = (int)A.size1(), size2 = (int)A.size2(), lwork = -1, m, num = iu-il+1;
        if (il < 1 || iu > size1 || iu < il) { throw std::runtime_error("heevx(A), the range (il,iu) is wrong.\n");}
        size_t info;
        if (size1 != size2) { throw std::runtime_error("heevx(A), A must be a square matrix.\n");}
        S.resize(size1);
        V.resize({A.size1(), size_t(num)});
        T vl ,vu;
        std::vector<T> work;
        std::vector<int> ifail(size1), iwork(5*size1);
        T wkopt;
        info = boost::numeric::bindings::lapack::detail::heevx('V', 'I', boost::numeric::bindings::tag::lower(),
         size1, A.data(), size1, vl, vu, il, iu, abstol, m, S.data(), V.data(), size1, &wkopt, lwork, iwork.data(), ifail.data());
        lwork = (int)wkopt;
        work.resize(lwork);
        info = boost::numeric::bindings::lapack::detail::heevx('V', 'I', boost::numeric::bindings::tag::lower(),
         size1, A.data(), size1, vl, vu, il, iu, abstol, m, S.data(), V.data(), size1, work.data(), lwork, iwork.data(), ifail.data());
        return info;
    }
    
    
    template<typename T>
    typename std::enable_if<boost::is_complex<T>::value, size_t>::type
    heevx(gtensor<T, 2> & A, diagmatrix<typename boost::numeric::bindings::traits::type_traits<T>::real_type> & S, gtensor<T, 2> & V, int il, int iu, typename boost::numeric::bindings::traits::type_traits<T>::real_type abstol = 1.0e-12)
    {
        if (A.empty()) { throw std::runtime_error("heevx(A), A is empty.\n");}
        int size1 = (int)A.size1(), size2 = (int)A.size2(), lwork = -1, m, num = iu-il+1;
        if (il < 1 || iu > size1 || iu < il) { throw std::runtime_error("heevx(A), the range (il,iu) is wrong.\n");}
        size_t info;
        if (size1 != size2) { throw std::runtime_error("heevx(A), A must be a square matrix.\n");}
        S.resize(size1);
        V.resize({A.size1(), size_t(num)});
        typename boost::numeric::bindings::traits::type_traits<T>::real_type vl ,vu;
        std::vector<T> work;
        std::vector<typename boost::numeric::bindings::traits::type_traits<T>::real_type> rwork(7*size1);
        std::vector<int> ifail(size1), iwork(5*size1);
        T wkopt;
        info = boost::numeric::bindings::lapack::detail::heevx('V', 'I', boost::numeric::bindings::tag::lower(), size1, A.data(), size1, vl, vu, il, iu, abstol, m, S.data(), V.data(), size1, &wkopt, lwork, rwork.data(), iwork.data(), ifail.data());
        lwork = (int)std::real(wkopt);
        work.resize(lwork);
        info = boost::numeric::bindings::lapack::detail::heevx('V', 'I', boost::numeric::bindings::tag::lower(), size1, A.data(), size1, vl, vu, il, iu, abstol, m, S.data(), V.data(), size1, work.data(), lwork, rwork.data(), iwork.data(), ifail.data());
        return info;
    }
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value, size_t>::type
    geev(const char jobvl, const char jobvr, gtensor<T, 2> & A, diagmatrix<T> & wr, diagmatrix<T> & wi, gtensor<T, 2> & vl, gtensor<T, 2> & vr )
    {
        if (A.empty()) { throw std::runtime_error("eig(A), A is empty.\n");}
#ifdef GTENSOR_DEBUG
        assert(A.size1() == A.size2());
#endif
        std::array<size_t, 2> dim = A.dim();
        if (jobvl=='V') { vl.resize(dim);}
        if (jobvr=='V') { vr.resize(dim);}
        int lwork = -1, N = (int)dim[0];
        wr.resize(N), wi.resize(N);
        T wkopt;
        size_t info = boost::numeric::bindings::lapack::detail::geev(jobvl, jobvr, N, A.data(), N, wr.data(), wi.data(), vl.data(), N, vr.data(), N, &wkopt, lwork);
        lwork = (int)std::real(wkopt);
        std::vector<T> work(lwork);
        info = boost::numeric::bindings::lapack::detail::geev(jobvl, jobvr, N, A.data(), N, wr.data(), wi.data(), vl.data(), N, vr.data(), N, work.data(), lwork);
        return info;
    }
    
    
    template<typename T>
    typename std::enable_if<boost::is_complex<T>::value, size_t>::type
    geev(const char jobvl, const char jobvr, gtensor<T, 2> & A, diagmatrix<T> &  w, gtensor<T, 2> & vl, gtensor<T, 2> & vr )
    {
        if (A.empty()) { throw std::runtime_error("eig(A), A is empty.\n");}
#ifdef GTENSOR_DEBUG
        assert(A.size1() == A.size2());
#endif
        std::array<size_t, 2> dim = A.dim();
        if (jobvl=='V') { vl.resize(dim);}
        if (jobvr=='V') { vr.resize(dim);}
        int lwork = -1, N = (int)dim[0];
        w.resize(N);
        T wkopt;
        std::vector<typename boost::numeric::bindings::traits::type_traits<T>::real_type> rwork(2*N);
        size_t info = boost::numeric::bindings::lapack::detail::geev(jobvl, jobvr, N, A.data(), N, w.data(), vl.data(), N, vr.data(), N, &wkopt, lwork, rwork.data());
        lwork = (int)std::real(wkopt);
        std::vector<T> work(lwork);
        info = boost::numeric::bindings::lapack::detail::geev(jobvl, jobvr, N, A.data(), N, w.data(), vl.data(), N, vr.data(), N, work.data(), lwork, rwork.data());
        return info;
    }
    
    template<typename T>
    typename std::enable_if<boost::is_scalar<T>::value, size_t>::type
    geev(const char jobvl, const char jobvr, gtensor<T, 2> & A, diagmatrix<std::complex<T> > &  w, gtensor<std::complex<T>, 2> & vl, gtensor<std::complex<T>, 2> & vr )
    {
        if (A.empty()) { throw std::runtime_error("eig(A), A is empty.\n");}
#ifdef GTENSOR_DEBUG
        assert(A.size1() == A.size2());
#endif
        gtensor<std::complex<T>, 2> Ac(A);
        return geev(jobvl, jobvr, Ac, w, vl, vr);
    }
    
    
    
    template <typename T>
    void expm_padm(gtensor<T, 2> & A, gtensor<T, 2> & A2, gtensor<T, 2> & I, gtensor<T, 2> &P, gtensor<T, 2> &Q, int p)
    {
        typedef typename boost::numeric::bindings::traits::type_traits<T>::real_type magnitude_type;
        std::vector<T> c(p+1);
        int i, j, N = (int)A.size1();
        size_t info;
        T alpha = 1.0, beta = 0.0;
        c[0] = 1;
        for (i=1; i<p+1; ++i) {
            c[i] = c[i-1]*(T(p+1-i)/T(i*(2*p+1-i)));
        }
        i = (int)boost::numeric::bindings::blas::detail::iamax((int)A.size(), A.data(), 1);
        magnitude_type s = std::abs(A[i-1]);
//        std::cout << "s: " << s << std::endl;
        if (s>0.5) {
//            std::cout << "s: " << s << std::endl;
            s = std::max(0, int(log(s)/log(2.0))+2);
            A *= pow(2.0, -s);
        }
        I = A;
//        boost::numeric::bindings::blas::detail::gemm('N', 'N', N, N, N, alpha, I.data(), N, A.data(), N, beta, A2.data(), N);
        gemm(alpha, I, A, beta, A2);
//        gemm(I, A, A2);
        std::vector<int> ipiv(N);
        P = 0.0;
        Q = 0.0;
        std::array<size_t, 2> index;
        for (i=0; i<N; ++i) {
            index[0] = i, index[1] = i;
            P(index) = c[p-1];
            Q(index) = c[p];
        }
        int odd = 1;
        for (i=p-2; i>=0; --i) {
            if (odd==1) {
//                boost::numeric::bindings::blas::detail::gemm('N', 'N', N, N, N, alpha, Q.data(), N, A2.data(), N, beta, I.data(), N);
                gemm(alpha, Q, A2, beta, I);
                for (j=0; j<N; ++j) {
                    index[0] = j, index[1] = j;
                    I(index)+=c[i];
                }
                swap(Q, I);
            }
            else{
//                boost::numeric::bindings::blas::detail::gemm('N', 'N', N, N, N, alpha, P.data(), N, A2.data(), N, beta, I.data(), N);
                gemm(alpha, P, A2, beta, I);
                for (j=0; j<N; ++j) {
                    index[0] = j, index[1] = j;
                    I(index)+=c[i];
                }
                swap(P, I);
                
            }
            odd = 1-odd;
        }
        swap(A2, A);
        if (odd==1) {
//            boost::numeric::bindings::blas::detail::gemm('N', 'N', N, N, N, alpha, Q.data(), N, A2.data(), N, beta, I.data(), N);
            gemm(alpha, Q, A2, beta, I);
            I-=P;
            info = gesv(I, P, ipiv);
            if (info != 0) { throw std::runtime_error("gesv(a, b)(b/a) falis.\n");}
            swap(A, P);
            A*=2.0;
            for (i=0; i<N; ++i) {
                index[0] = i, index[1] = i;
                A(index)-=1.0;
            }
        }
        else{
//            boost::numeric::bindings::blas::detail::gemm('N', 'N', N, N, N, alpha, P.data(), N, A2.data(), N, beta, I.data(), N);
            gemm(alpha, P, A2, beta, I);
            Q-=I;
            info = gesv(Q, I, ipiv);
            if (info != 0) { throw std::runtime_error("gesv(a, b)(b/a) falis.\n");}
            swap(A, I);
            A*=2.0;
            for (i=0; i<N; ++i) {
                index[0] = i, index[1] = i;
                A(index)+=1.0;
            }
        }
        for (i=0; i<int(s); ++i) {
            P = A;
            gemm(alpha, A, P, beta, I);
//            boost::numeric::bindings::blas::detail::gemm('N', 'N', N, N, N, alpha, A.data(), N, P.data(), N, beta, I.data(), N);
            swap(I, A);
        }
    }
    
    
    template <typename T>
    inline void expm_padm(gtensor<T, 2> & A, int p = 6)
    {
        assert(A.size1() == A.size2());
        std::array<size_t, 2> size;
        size[0] = A.size1(), size[1] = A.size2();
        gtensor<T, 2> I(size), A2(size), P(size), Q(size);
        expm_padm(A, A2, I, P, Q, p);
    }


    
}




#endif
