//
//  gtensor_algorithm.h
//  gtensor
//
//  Created by guochu on 20/11/15.
//  Copyright © 2015 guochu. All rights reserved.
//

#ifndef guochu_gtensor_gtensor_blas1_h
#define guochu_gtensor_gtensor_blas1_h


#include "gtensor.h"
#include <vector>
#include <map>


namespace guochu {
    
    /**********************************************************
     some nonmember functions.
     *********************************************************/
    
    
    template <typename T1, typename T2, size_t N, typename major>
    gtensor<T1, N, major>
    operator*(const gtensor<T1, N, major> & A, const diagmatrix<T2> & B)
    {
        gtensor<T1, N, major> C(A);
        C.colscale(B);
        return C;
    }
    
    template <typename T1, typename T2, size_t N, typename major>
    gtensor<T1, N, major>
    operator*(const diagmatrix<T2> & B, const gtensor<T1, N, major> & A)
    {
        gtensor<T1, N, major> C(A);
        C.rowscale(B);
        return C;
    }
    
    template<typename T>
    diagmatrix<T> operator*(const diagmatrix<T> & A, T value)
    {
        diagmatrix<T> B(A.size());
        for (size_t i = 0; i < B.size(); ++i) {
            B[i] = A[i]*value;
        }
        return B;
    }
    
    template<typename T>
    diagmatrix<T> operator/(const diagmatrix<T> & A, T value)
    {
        diagmatrix<T> B(A.size());
        for (size_t i = 0; i < B.size(); ++i) {
            B[i] = A[i]/value;
        }
        return B;
    }

    
    template<typename T>
    diagmatrix<T> operator*(T value, const diagmatrix<T> & A)
    {
        diagmatrix<T> B(A.size());
        for (size_t i = 0; i < B.size(); ++i) {
            B[i] = A[i]*value;
        }
        return B;
    }


    
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator*(const tensor_expression<Ten> & A, typename Ten::value_type alpha)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)*alpha;}
        return C;
    }
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator/(const tensor_expression<Ten> & A, typename Ten::value_type alpha)
    {
        BOOST_STATIC_ASSERT(std::is_same<typename Ten::trans_type, label::no_transpose>::value);
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)/alpha;}
        return C;
    }
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator*( typename Ten::value_type alpha, const tensor_expression<Ten> & A)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)*alpha;}
        return C;
    }
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator+(const tensor_expression<Ten> & A, typename Ten::value_type alpha)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)+alpha;}
        return C;
    }
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator+(typename Ten::value_type alpha, const tensor_expression<Ten> & A)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)+alpha;}
        return C;
    }
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator-(const tensor_expression<Ten> & A, typename Ten::value_type alpha)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)-alpha;}
        return C;
    }
    
    template <typename Ten >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator-(typename Ten::value_type alpha, const tensor_expression<Ten> & A)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = alpha - A().operator[](i);}
        return C;
    }

    
    template <typename Ten, typename TenB >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator+(const tensor_expression<Ten> & A, const tensor_expression<TenB> & B)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        assert(A().dim() == B().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)+B().operator[](i);}
        return C;
    }
    
    template <typename Ten, typename TenB >
    gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    operator-(const tensor_expression<Ten> & A, const tensor_expression<TenB> & B)
    {
        BOOST_STATIC_ASSERT(std::is_same<typename Ten::orientation_type, typename TenB::orientation_type>::value);
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> C(A().dim());
        assert(A().dim() == B().dim());
        for (size_t i=0; i<A().size(); ++i) {C[i] = A().operator[](i)-B().operator[](i);}
        return C;
    }
    
    

    template <typename Ten, typename TenB>
    void conj(const tensor_expression<Ten> & A, tensor_expression<TenB> & B, typename Ten::value_type sca = 1.0)
    {
        BOOST_STATIC_ASSERT(std::is_same<typename Ten::orientation_type, typename TenB::orientation_type>::value);
        assert(A().dim() == B().dim());
        if (sca == 1.0) {
            for (size_t i = 0; i < A().size(); ++i) {B().operator[](i) = conj(A().operator[](i));}
        }
        else if (sca == 0.0){
            for (size_t i = 0; i < A().size(); ++i) {B().operator[](i) = 0.0;}
        }
        else{
            for (size_t i = 0; i < A().size(); ++i) {B().operator[](i) = conj(A().operator[](i))*sca;}
        }
    }
    
    
    template <typename Ten>
    inline gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type>
    conj(const tensor_expression<Ten> & A)
    {
        gtensor<typename Ten::value_type, std::tuple_size<typename Ten::dimvector_type>::value, typename Ten::orientation_type> B(A().dim());
        conj(A, B, 1.0);
        return B;
    }
    

    
    template<typename T, size_t N, typename major>
    gtensor<T, N, major>
    trans(const gtensor<T, N, major> & A)
    {
        gtensor<T, N, major> B(A.trans());
        return B;
    }
    
    
    template<typename T, size_t N, typename major>
    gtensor<T, N, major>
    herm(const gtensor<T, N, major> & A)
    {
        gtensor<T, N, major> B(A.herm());
        return B;
    }


    
    
    


    


    
}

#endif /* gtensor_algorithm_h */


