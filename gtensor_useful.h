
#ifndef guochu_gtensor_gtensor_useful_h
#define guochu_gtensor_gtensor_useful_h

#include "gtensor.h"
#include "tensor_index.h"

namespace guochu {
    
    
    // hard copy, tensor a and b should have the same size, but they do not need to have the same shape.
    template<typename T1, size_t N1, typename T2, size_t N2, typename major>
    void copy(const gtensor<T1, N1, major> & a, gtensor<T2, N2, major> & b)
    {
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            b[i] = a[i];
        }
    }

    template<typename Ten1, typename Ten2>
    gtensor<typename Ten1::value_type, std::tuple_size<typename Ten1::dimvector_type>::value, typename Ten1::orientation_type>
    cat_gtensor(const tensor_expression<Ten1> & t1, const tensor_expression<Ten2> & t2, size_t axis)
    {
        BOOST_STATIC_ASSERT(std::is_same<typename Ten1::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::is_same<typename Ten2::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::tuple_size<typename Ten1::dimvector_type>::value == std::tuple_size<typename Ten2::dimvector_type>::value);
        BOOST_STATIC_ASSERT(std::is_same<typename Ten1::orientation_type, typename Ten2::orientation_type>::value);
        typename Ten1::dimvector_type dim1 = t1().dim(), dim2 = t2().dim(), dim3, dimstart, dimstart1;
        axis -= 1;
        for (size_t i = 0; i < dim1.size(); ++i) {
            if (i == axis) {
                dim3[i] = dim1[i] + dim2[i];
                dimstart1[i] = dim1[i];
            }
            else{
                if (dim1[i] != dim2[i]) {
                    throw std::runtime_error("cat_gtensor(A,B), the size of A and B does not match.\n");
                }
                dimstart1[i] = 0;
                dim3[i] = dim1[i];
            }
        }
        gtensor<typename Ten1::value_type, std::tuple_size<typename Ten1::dimvector_type>::value, typename Ten1::orientation_type> t3(dim3);
        dimstart.fill(0);
        t3(dimstart, dim1) = t1;
        t3(dimstart1, dim3) = t2;
        return t3;
    }
    
    template<typename Ten1, typename Ten2>
    gtensor<typename Ten1::value_type, std::tuple_size<typename Ten1::dimvector_type>::value, typename Ten1::orientation_type>
    diag(const tensor_expression<Ten1> & t1, const tensor_expression<Ten2> & t2)
    {
        BOOST_STATIC_ASSERT(std::is_same<typename Ten1::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::is_same<typename Ten2::trans_type, label::no_transpose>::value);
        BOOST_STATIC_ASSERT(std::tuple_size<typename Ten1::dimvector_type>::value == std::tuple_size<typename Ten2::dimvector_type>::value);
        BOOST_STATIC_ASSERT(std::is_same<typename Ten1::orientation_type, typename Ten2::orientation_type>::value);
        
        tensor::tensor_index<std::tuple_size<typename Ten1::dimvector_type>::value> dim = t1().dim() + t2().dim(), index;
        gtensor<typename Ten1::value_type, std::tuple_size<typename Ten1::dimvector_type>::value, typename Ten1::orientation_type> t3(dim, 0);
        index.fill(0);
        t3(index, t1().dim()) = t1;
        t3(t1.dim(), t3.dim()) = t2;
        return t3;
    }

    
    

    
    /*********************************************************************************
     the localbase serves as a table to map the mapped_matrix type to a dense matrix
     ********************************************************************************/
    template<typename Index1, typename Index2, typename T>
    typename std::enable_if<std::is_scalar<T>::value||boost::is_complex<T>::value, gtensor<T, 2> >::type
    Smat2Dmat(const std::map<Index2, std::map<Index2, T> > & smat, const std::vector<Index1> & basisl, const std::vector<Index2> & basisr)
    {
        typename std::map<Index2, std::map<Index2, T> >::const_iterator itr1;
        typename std::map<Index2, T>::const_iterator itr2;
        size_t sizel = basisl.size(), sizer = basisr.size(), i, j;
        gtensor<T, 2> dmat({sizel, sizer}, 0.0);
        for (i = 0 ; i<sizel; ++i) {
            itr1 = smat.find(basisl[i]);
            if (itr1 != smat.cend()) {
                for (j = 0; j<sizer; ++j) {
                    itr2 = itr1->second.find(basisr[j]);
                    if (itr2 != itr1->second.cend()) {
                        dmat({i,j}) = itr2->second;
                    }
                }
            }
        }
        return dmat;
    }
    
    
    template<typename Index1, typename Index2, typename T>
    gtensor<T, 2>
    Smat2Dmat(const std::map<Index2, std::map<Index2, gtensor<T, 2> > > & smat, const std::vector<Index1> & basisl, const std::vector<Index2> & basisr)
    {
        typename std::map<Index2, std::map<Index2, T> >::const_iterator itr1;
        typename std::map<Index2, T>::const_iterator itr2;
        size_t sizel = basisl.size(), sizer = basisr.size(), i, j;
        std::vector<size_t> basissizel(sizel, 0), basissizer(sizer, 0), sizevecl(sizel+1), sizevecr(sizer+1);
        sizevecl[0] = 0, sizevecr[0] = 0;
        for (i = 0; i < basisl.size(); ++i) {
            itr1 = smat.find(basisl[i]);
            if (itr1 != smat.cend()) {
                itr2 = itr1->second.cbegin();
                basissizel[i] = itr2->second.size1();
#ifdef SHIFU_DEBUG
                for (; itr2 != itr1->second.cend(); ++itr2) {
                    if (itr2->second.size1() != basissizel[i]) {
                        throw std::runtime_error("Smat2Dmat(block), the first size does not match.\n");
                    }
                }
#endif
                for (j = 0; i < basisr.size(); ++i) {
                    itr2 = itr1->second.find(basisr[j]);
                    if (itr2 != itr1->second.cend()) {
                        if (basissizer[j] == 0) {
                            basissizer[j] = itr2->second.size2();
                        }
                        else{
                            if (basissizer[j] != itr2->second.size2()) {
                                throw std::runtime_error("Smat2Dmat(block), the second size does not match.\n");
                            }
                        }
                    }
                }
            }
        }
        for (i = 0; i < basissizel.size(); ++i) {
            sizevecl[i+1] = basissizel[i] + sizevecl[i];
        }
        for (i = 0; i < basissizer.size(); ++i) {
            sizevecr[i+1] = basissizer[i] + sizevecr[i];
        }
        gtensor<T, 2> dmat({sizevecl.back(), sizevecr.back()}, 0.0);
        for (i = 0 ; i<sizel; ++i) {
            itr1 = smat.find(basisl[i]);
            if (itr1 != smat.cend()) {
                for (j = 0; j<sizer; ++j) {
                    itr2 = itr1->second.find(basisr[j]);
                    if (itr2 != itr1->second.cend()) {
                        dmat({sizevecl[i],sizevecr[j]},{sizevecl[i+1], sizevecr[j+1]}) = itr2->second;
                    }
                }
            }
        }
        return dmat;
    }
    
    
    
    
    template <typename T>
    std::pair<size_t, typename boost::numeric::bindings::traits::type_traits<T>::real_type>
    cutoff(diagmatrix<typename boost::numeric::bindings::traits::type_traits<T>::real_type> & S, gtensor<T, 2> & U, gtensor<T, 2> & V, int maxbonddimension, typename boost::numeric::bindings::traits::type_traits<T>::real_type svdcutoff)
    {
        assert(maxbonddimension > 0);
        std::pair<size_t, typename boost::numeric::bindings::traits::type_traits<T>::real_type> bonderror(0,0.);
        size_t size1, size2, size3;
        size1 = U.size1();
        size2 = S.size();
        size3 = V.size2();
        size_t dim = size2;
        for (size_t i=0; i<size2; ++i) {
            if (S[i]<svdcutoff) {
                dim=i;
                break;
            }
        }
        bonderror.first = dim;
        bonderror.second = -1.;
        if (dim > maxbonddimension) {
#ifdef SHIFU_VERBOSE
            std::cout << "sum: " << size2 << " -> "<<dim << " (exceeds the max bonddimension "<< maxbonddimension << ", cut off error is: "<<S[maxbonddimension]<<")"<< std::endl;
#endif
            bonderror.second = S[maxbonddimension];
            U.resize({size1, static_cast<size_t>(maxbonddimension)}, true);
            S.resize(maxbonddimension, true);
            V.resize({static_cast<size_t>(maxbonddimension), size3}, true);
            bonderror.first = maxbonddimension;
            return bonderror;
        }
        if (dim==size2) {
#ifdef SHIFU_VERBOSE
            std::cout << "sum: " << size2 << " -> "<<dim << std::endl;
#endif
            return bonderror;}
        if (dim==0) {
#ifdef SHIFU_DEBUG
            printf("the bonddimension is zeros after cutoff.\n");
#endif
        }
        bonderror.second = S[dim];
        U.resize({size1, dim}, true);
        S.resize(dim, true);
        V.resize({dim, size3}, true);
        bonderror.first = dim;
#ifdef SHIFU_VERBOSE
        std::cout << "sum: " << size2 << " -> "<<dim << std::endl;
#endif
        return bonderror;
    }
    
    template<typename T, size_t N, typename major>
    bool is_zero(const gtensor<T, N, major> & A)
    {
        typename gtensor<T, N, major>::value_type zero = 0.;
        for (size_t i = 0; i < A.size(); ++i) {
            if (A[i] != zero) {
                return false;
            }
        }
        return true;
    }
    
    template<typename T>
    typename std::enable_if<std::is_scalar<T>::value || boost::is_complex<T>::value, bool>::type
    is_zero(T value)
    {
        T zero = 0.0;
        if (value != zero) {
            return false;
        }
        return true;
    }
    
    
    template<typename T, typename major>
    T trace(const gtensor<T, 2, major> & mat)
    {
        typename gtensor<T, 2, major>::dimvector_type ind;
        if (mat.size1() != mat.size2()) {
            throw std::runtime_error("can not get trace of non square matrix.\n");
        }
        T tr = 0.;
        for (size_t i = 0; i < mat.size1(); ++i) {
            ind[0] = i, ind[1] = i;
            tr += mat(ind);
        }
        return tr;
    }
    
    template<typename T, typename major>
    diagmatrix<T> diag(const gtensor<T, 2, major> & mat, int line = 0)
    {
        assert(mat.size1() == mat.size2());
#ifdef GTENSOR_DEBUG
        assert(!mat.empty());
        assert(std::abs(line) < mat.size1());
#endif
        size_t L = mat.size1() - std::abs(line), i;
        diagmatrix<T> vec(L);
        for (i = 0; i < L; i++) {
            vec[i] = mat(i, i + line);
        }
        return vec;
    }
    
    
    template<typename T>
    T mean(const std::vector<T> & vec)
    {
        return tensor::mean(vec.data(), vec.size());
    }
    
    template<typename T>
    T stdvar(const std::vector<T> & vec)
    {
        return tensor::stdvar(vec.data(), vec.size());
    }

    
}



#endif

