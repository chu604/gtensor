//
//  gtensor.h
//  gtensor
//
//  Created by guochu on 19/11/15.
//  Copyright © 2015 guochu. All rights reserved.
//

#ifndef guochu_gtensor_gtensor_h
#define guochu_gtensor_gtensor_h

#include "tensor_expression.h"
#include <iostream>
#include <valarray>
#include <boost/type_traits/is_complex.hpp>
#include "gtensor_utility.h"
#include "../num_traits.h"
#include <stdexcept>
#include <boost/serialization/complex.hpp>
#include "diagmatrix.h"
#include "matrix_transpose.h"
#include "tensor_block_view.h"
#include "tensor_trans_view.h"
#include "tensor_index.h"

namespace guochu {
    
    template<typename T, size_t N>
    void print(const std::array<T, N> & vec)
    {
        return tensor::print_vector(vec.data(), N);
    }
    
    template <typename T, size_t N, typename major = label::column_major>
    class gtensor :
    public tensor_expression<gtensor<T, N, typename std::enable_if<N !=0, major>::type> >  {
        typedef gtensor<T, N, major> self_type;
        typedef tensor::tensor_block_view<T, N, major> block_view_type;
        typedef tensor::tensor_block_const_view<T, N, major> block_const_view_type;
    public:
        typedef size_t size_type;
        typedef major orientation_type;
        typedef label::no_transpose trans_type;
        typedef T value_type;
        typedef value_type scalar_type;
        typedef self_type vector_type;
        typedef typename traits::num_traits<value_type>::magnitude_type magnitude_type;
        typedef std::array<size_type, N>                dimvector_type;
        typedef std::valarray<T>                        data_type;
        
        //constructors
        gtensor();
        explicit gtensor(const dimvector_type & vec);
        explicit gtensor(size_t size1);
        explicit gtensor(size_t size1, size_t size2);
        explicit gtensor(size_t size1, size_t size2, size_t size3);
        explicit gtensor(size_t size1, size_t size2, size_t size3, size_t size4);
        gtensor(const dimvector_type & vec, value_type value);
        gtensor(const value_type *value, const dimvector_type & vec);
        
            
        template <typename E>
        gtensor(const tensor_expression<E>& A);
        
        template <typename T1, typename major1>
        gtensor(const tensor::tensor_block_view<T1, N, major1>& view1);
        
        template <typename T1, typename major1>
        gtensor(const tensor::tensor_block_const_view<T1, N, major1>& view1);
        
        template <typename DT, typename TR, typename major1>
        gtensor(const tensor::matrix_trans_const_view<DT, TR, major1> & A);
        
        value_type *data(){return &data_[0];}
        const value_type *data()const{return &data_[0];}
            
        //value reference
        inline value_type &operator()(const dimvector_type & vec){
#ifdef GTENSOR_DEBUG
            for (size_t i = 0; i < N; ++i) {
                assert(vec[i] < dim_[i]);
            }
#endif
            orientation_type maj;
            return data_[tensor::mind2sind<N>(vec, cudim_, maj)];}
        inline const value_type &operator()(const dimvector_type & vec)const{
#ifdef GTENSOR_DEBUG
            for (size_t i = 0; i < N; ++i) {
                assert(vec[i] < dim_[i]);
            }
#endif
            orientation_type maj;
            return data_[tensor::mind2sind<N>(vec, cudim_, maj)];}
        
        inline value_type &operator()(size_t size1){
            BOOST_STATIC_ASSERT(N==1);
            assert(size1 < dim_[0]);
            return this->operator[](size1);
        }
        
        inline const value_type &operator()(size_t size1)const{
            BOOST_STATIC_ASSERT(N==1);
            assert(size1 < dim_[0]);
            return this->operator[](size1);
        }

        inline value_type &operator()(size_t size1, size_t size2){
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1]);
            return this->operator[](tensor::mind2sind(size1, size2, cudim_, ori));
        }
        
        inline const value_type &operator()(size_t size1, size_t size2)const{
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1]);
            return this->operator[](tensor::mind2sind(size1, size2, cudim_, ori));
        }
        
        inline value_type &operator()(size_t size1, size_t size2, size_t size3){
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, cudim_, ori));
        }
        
        inline const value_type &operator()(size_t size1, size_t size2, size_t size3)const{
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, cudim_, ori));
        }
        
        inline value_type &operator()(size_t size1, size_t size2, size_t size3, size_t size4){
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2] && size4 < dim_[3]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, size4, cudim_, ori));
        }
        
        inline const value_type &operator()(size_t size1, size_t size2, size_t size3, size_t size4)const{
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2] && size4 < dim_[3]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, size4, cudim_, ori));
        }
        
        inline value_type &operator()(size_t size1, size_t size2, size_t size3, size_t size4, size_t size5){
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2] && size4 < dim_[3] && size5 < dim_[4]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, size4, size5, cudim_, ori));
        }
        
        inline const value_type &operator()(size_t size1, size_t size2, size_t size3, size_t size4, size_t size5)const{
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2] && size4 < dim_[3] && size5 < dim_[4]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, size4, size5, cudim_, ori));
        }
        
        inline value_type &operator()(size_t size1, size_t size2, size_t size3, size_t size4, size_t size5, size_t size6){
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2] && size4 < dim_[3] && size5 < dim_[4] && size6 < dim_[5]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, size4, size5, size6, cudim_, ori));
        }
        
        inline const value_type &operator()(size_t size1, size_t size2, size_t size3, size_t size4, size_t size5, size_t size6)const{
            orientation_type ori;
            assert(size1 < dim_[0] && size2 < dim_[1] && size3 <dim_[2] && size4 < dim_[3] && size5 < dim_[4] && size6 < dim_[5]);
            return this->operator[](tensor::mind2sind(size1, size2, size3, size4, size5, size6, cudim_, ori));
        }
        
        inline value_type& operator[](size_type size1){
            assert(size1<=size_);
            return data_[size1];
        }
        
        inline const value_type& operator[](size_type size1)const{
            assert(size1<=size_);
            return data_[size1];
        }


        //            subtensor reference
        block_view_type operator()(const dimvector_type & startind, const dimvector_type & endind)
        {
            return block_view_type(data(), dim_, cudim_, startind, endind);
        }
        
        block_const_view_type operator()(const dimvector_type & startind, const dimvector_type & endind)const
        {
            return block_const_view_type(data(), dim_, cudim_, startind, endind);
        }
        
        block_view_type row(size_t row_)
        {
            BOOST_STATIC_ASSERT(N > 0);
            assert(row_ < size1());
            dimvector_type start, end = dim_;
            start.fill(0), start[0] = row_, end[0] = row_+1;
            return block_view_type(data(), dim_, cudim_, start, end);
        }
        
        block_const_view_type row(size_t row_)const
        {
            BOOST_STATIC_ASSERT(N > 0);
            assert(row_ < size1());
            dimvector_type start, end = dim_;
            start.fill(0), start[0] = row_, end[0] = row_+1;
            return block_const_view_type(data(), dim_, cudim_, start, end);
        }
        
        
        block_view_type col(size_t col_)
        {
            BOOST_STATIC_ASSERT(N > 1);
            assert(col_ < sizeE());
            dimvector_type start, end = dim_;
            start.fill(0), start[N-1] = col_, end[N-1] = col_+1;
            return block_view_type(data(), dim_, cudim_, start, end);
        }
        
        block_const_view_type col(size_t col_)const
        {
            BOOST_STATIC_ASSERT(N > 1);
            assert(col_ < sizeE());
            dimvector_type start, end = dim_;
            start.fill(0), start[N-1] = col_, end[N-1] = col_+1;
            return block_const_view_type(data(), dim_, cudim_, start, end);
        }
        
        
        const self_type &operator=(scalar_type A)
        {
            data_ = A;
            return *this;
        }
        
        template <typename E>
        const self_type &operator=(const tensor_expression<E> & A);
        
        template <typename T1, typename major1>
        const self_type &operator=(const tensor::tensor_block_view<T1, N, major1>& view1);
        
        template <typename T1, typename major1>
        const self_type &operator=(const tensor::tensor_block_const_view<T1, N, major1>& view1);
        
        template <typename DT, typename TR, typename major1>
        const self_type &operator=(const tensor::matrix_trans_const_view<DT, TR, major1> & A);
        


        template <typename E>
        const self_type &operator+=(const tensor_expression<E> & A);
        const self_type &operator+=(const self_type & A);
        template <typename E>
        const self_type &operator-=(const tensor_expression<E> & A);
        const self_type &operator-=(const self_type & A);
        const self_type &operator+=(value_type A);
        const self_type &operator-=(value_type A);
        const self_type &operator*=(value_type alpha);
        const self_type &operator/=(value_type alpha);
        template<typename T1>
        typename std::enable_if<std::is_scalar<T1>::value || boost::is_complex<T1>::value, const self_type>::type
        &rowscale(const diagmatrix<T1> & vec);
        template<typename T1>
        typename std::enable_if<std::is_scalar<T1>::value || boost::is_complex<T1>::value, const self_type>::type
        &colscale(const diagmatrix<T1> & vec);

        
        template <typename E>
        inline const self_type &operator*=(const diagmatrix<E> & diagmat){return this->colscale(diagmat);}
            
        void swap(self_type & A);
        
        tensor::matrix_trans_const_view<T, label::transpose, major> trans()const
        {
            BOOST_STATIC_ASSERT(N==2);
            return tensor::matrix_trans_const_view<T, label::transpose, major>(data(), dim_[0], dim_[1]);
        }
        
        tensor::matrix_trans_const_view<T, label::conjugate, major> herm()const
        {
            BOOST_STATIC_ASSERT(N==2);
            return tensor::matrix_trans_const_view<T, label::conjugate, major>(data(), dim_[0], dim_[1]);
        }
        
        tensor::matrix_trans_const_view<T, label::transpose, major> trans(size_t i, size_t j)const
        {
            assert(i*j == size_);
            return tensor::matrix_trans_const_view<T, label::transpose, major>(data(), i, j);
        }
        
        tensor::matrix_trans_const_view<T, label::conjugate, major> herm(size_t i, size_t j)const
        {
            assert(i*j == size_);
            return tensor::matrix_trans_const_view<T, label::conjugate, major>(data(), i, j);}
        
        tensor::matrix_trans_const_view<T, label::no_transpose, major> matrix(size_t i, size_t j)const
        {
            assert(i*j == size_);
            return tensor::matrix_trans_const_view<T, label::no_transpose, major>(data(), i, j);
        }

        void set_subtensor(const value_type *subtensor, const dimvector_type & startvec, const dimvector_type & endvec);
        inline void set_row(const value_type* value, size_t i){return this->set_subtensor(value, {i, 0},{i+1, dim_[1]});}
        inline void set_col(const value_type* value, size_t i){return this->set_subtensor(value, {0, i},{dim_[0], i+1});}

        
        void write()const;
        void reshape(const dimvector_type & vec);
        
        void resize(const dimvector_type & vec, bool keep = false);
        
        void clear(){size_ = 0, data_.resize(0); dim_.fill(0), cudim_.fill(0);}
        bool empty()const{
            if (size_==0) { return true;}
            return false;
        }
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & size_;
            ar & dim_;
            ar & cudim_;
            ar & data_;
        }
            
            
        const dimvector_type& dim()const{return dim_;}
        const dimvector_type& cudim()const{return cudim_;}
        size_type size()const{return size_;}
        size_type rank()const{return N;}
        size_type leadingdim()const{BOOST_STATIC_ASSERT(N==2);
            return std::is_same<orientation_type, label::column_major>::value?dim_[0]:dim_[1];};
        size_type size1()const{BOOST_STATIC_ASSERT(N>0);return dim_[0];};
        size_type size2()const{BOOST_STATIC_ASSERT(N>1);return dim_[1];};
        size_type size3()const{BOOST_STATIC_ASSERT(N>2);return dim_[2];};
        size_type size4()const{BOOST_STATIC_ASSERT(N>3);return dim_[3];};
        size_type size5()const{BOOST_STATIC_ASSERT(N>4);return dim_[4];};
        size_type size6()const{BOOST_STATIC_ASSERT(N>5);return dim_[5];};
        size_type size7()const{BOOST_STATIC_ASSERT(N>6);return dim_[6];};
        size_type size8()const{BOOST_STATIC_ASSERT(N>7);return dim_[7];};
        size_type sizeE()const{BOOST_STATIC_ASSERT(N>0);return dim_[N-1];}
        
        
        template<size_t N2, typename T1, size_t N1, typename maj>
        friend gtensor<T1, N2, maj> reshape(gtensor<T1, N1, maj> & A, std::array<size_t, N2> dim);


    private:
        size_type           size_;
        dimvector_type      dim_;
        dimvector_type      cudim_;
        data_type           data_;
    };
    
    
    /*******************************************************
     constructors of the tensor class.
     ******************************************************/
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor()
    : tensor_expression<self_type>(), size_(0)
    {
        dim_.fill(0);
        cudim_.fill(0);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(const dimvector_type & vec)
    : tensor_expression<self_type>(), dim_(vec)
    {
        orientation_type maj;
        tensor::dim2cudim<N>(dim_, cudim_, maj);
        size_ = tensor::prod<N>(dim_);
        data_ = data_type(size_);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(size_t size1)
    {
        orientation_type maj;
        BOOST_STATIC_ASSERT(N==1);
        dim_[0] = size1;
        size_ = size1;
        tensor::dim2cudim<N>(dim_, cudim_, maj);
        data_ = data_type(size_);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(size_t size1, size_t size2)
    {
        orientation_type maj;
        BOOST_STATIC_ASSERT(N==2);
        dim_[0] = size1, dim_[1] = size2;
        size_ = tensor::prod<N>(dim_);
        tensor::dim2cudim<N>(dim_, cudim_, maj);
        data_ = data_type(size_);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(size_t size1, size_t size2, size_t size3)
    {
        orientation_type maj;
        BOOST_STATIC_ASSERT(N==3);
        dim_[0] = size1, dim_[1] = size2, dim_[2] = size3;
        size_ = tensor::prod<N>(dim_);
        tensor::dim2cudim<N>(dim_, cudim_, maj);
        data_ = data_type(size_);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(size_t size1, size_t size2, size_t size3, size_t size4)
    {
        orientation_type maj;
        BOOST_STATIC_ASSERT(N==4);
        dim_[0] = size1, dim_[1] = size2, dim_[2] = size3, dim_[3] = size4;
        size_ = tensor::prod<N>(dim_);
        tensor::dim2cudim<N>(dim_, cudim_, maj);
        data_ = data_type(size_);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(const dimvector_type & vec, value_type value)
    : dim_(vec)
    {
        orientation_type maj;
        size_ = tensor::prod<N>(dim_);
        data_ = data_type(value, size_);
        tensor::dim2cudim<N>(dim_, cudim_, maj);
    }
    
    template <typename T, size_t N, typename major>
    gtensor<T, N, major>::gtensor(const value_type *value, const dimvector_type & vec)
    : gtensor(vec)
    {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value[i];
        }
    }
    
    template <typename T, size_t N, typename major>
    template <typename T1, typename major1>
    gtensor<T, N, major>::gtensor(const tensor::tensor_block_view<T1, N, major1>& view1): //tested!
    gtensor(view1.dim())
    {
        if (std::is_same<major, major1>::value) {
            for (size_t i = 0; i < view1.size(); ++i) {
                data_[i] = view1[i];
            }
        }
        else{ // this is a bad way to do.
            gtensor<T1, N, major1> tmp1(view1);
            gtensor<T, N, major> tmp2(tmp1);
            this->swap(tmp2);
        }
    }
    
    template <typename T, size_t N, typename major>
    template <typename T1, typename major1>
    gtensor<T, N, major>::gtensor(const tensor::tensor_block_const_view<T1, N, major1>& view1): //tested!
    gtensor(view1.dim())
    {
        if (std::is_same<major, major1>::value) {
            for (size_t i = 0; i < view1.size(); ++i) {
                data_[i] = view1[i];
            }
        }
        else{ // this is a bad way to do.
            gtensor<T1, N, major1> tmp1(view1);
            gtensor<T, N, major> tmp2(tmp1);
            this->swap(tmp2);
        }
    }
    
    
    template <typename T, size_t N, typename major>
    template <typename DT, typename TR, typename major1>
    gtensor<T, N, major>::gtensor(const tensor::matrix_trans_const_view<DT, TR, major1>& A) //tested!
    : gtensor(A.dim())
    {
        if (std::is_same<major, major1>::value) {
            major maj;
            TR tr;
            size_t row = std::is_same<TR, label::no_transpose>::value?A.size1():A.size2();
            size_t col = std::is_same<TR, label::no_transpose>::value?A.size2():A.size1();
            T alpha = 1.;
            tensor::omatcopy(maj, tr, row, col, alpha, A.data(), data());
        }
        else{
            gtensor<DT, N, major1> tmp1(A);
            gtensor<T, N, major> tmp2(tmp1);
            this->swap(tmp2);
        }
    }
    
    template <typename T, size_t N, typename major>
    template <typename E>
    gtensor<T, N, major>::gtensor(const tensor_expression<E>& A) //tested!
    : gtensor(A().dim())
    {
        typename E::orientation_type majA;
        orientation_type maj;
        tensor::copyd2d(A().data(), A().cudim(), majA, data(), cudim_, maj, size_);
    }
    
    
    template <typename T, size_t N, typename major>
    void gtensor<T, N, major>::set_subtensor(const value_type *value, const dimvector_type & startvec, const dimvector_type & endvec)
    {
        dimvector_type index, subcudim;
        tensor::detail::sub(endvec.data(), startvec.data(), index.data(), N);
        orientation_type maj;
        tensor::dim2cudim<N>(index, subcudim, maj);
        size_t i, mapi;
        for (i=0; i<tensor::prod<N>(index); ++i) {
            mapi = tensor::sindview2sind(cudim_, i, subcudim, startvec, maj);
            data_[mapi] = value[i];
        }
    }
    
    template <typename T, size_t N, typename major>
    template <typename E>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator=(const tensor_expression<E> & A) //tested!
    {
        if (size_ == A().size()) {
            typename E::orientation_type majA;
            orientation_type maj;
            dim_ = A().dim();
            cudim_ = A().cudim();
            tensor::copyd2d(A().data(), A().cudim().data(), majA, data(), cudim_.data(), maj, size_, N);
        }
        else{
            self_type tmp(A);
            this->swap(tmp);
        }
        return *this;
    }
    
    
    template <typename T, size_t N, typename major>
    template <typename T1, typename major1>
    const typename gtensor<T, N, major>::self_type&
    gtensor<T, N, major>::operator=(const tensor::tensor_block_view<T1, N, major1>& view1)
    {
        if (size_ == view1.size() && std::is_same<major, major1>::value) {
            dim_ = view1.dim();
            major maj;
            tensor::dim2cudim(dim_.data(), cudim_.data(), N, maj);
            for (size_t i = 0; i < view1.size(); ++i) {
                data_[i] = view1[i];
            }
        }
        else{
            gtensor<T, N, major> tmp(view1);
            this->swap(tmp);
        }
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    template <typename T1, typename major1>
    const typename gtensor<T, N, major>::self_type&
    gtensor<T, N, major>::operator=(const tensor::tensor_block_const_view<T1, N, major1>& view1)
    {
        if (size_ == view1.size() && std::is_same<major, major1>::value) {
            dim_ = view1.dim();
            major maj;
            tensor::dim2cudim(dim_.data(), cudim_.data(), N, maj);
            for (size_t i = 0; i < view1.size(); ++i) {
                data_[i] = view1[i];
            }
        }
        else{
            gtensor<T, N, major> tmp(view1);
            this->swap(tmp);
        }
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    template <typename DT, typename TR, typename major1>
    const typename gtensor<T, N, major>::self_type&
    gtensor<T, N, major>::operator=(const tensor::matrix_trans_const_view<DT, TR, major1> & A) //tested!
    {
        if (size_ == A.size() && std::is_same<major, major1>::value) {
            major maj;
            TR tr;
            dim_ = A.dim();
            tensor::dim2cudim(dim_.data(), cudim_.data(), N, maj);
            size_t row = std::is_same<TR, label::no_transpose>::value?A.size1():A.size2();
            size_t col = std::is_same<TR, label::no_transpose>::value?A.size2():A.size1();
            T alpha = 1.;
            tensor::omatcopy(maj, tr, row, col, alpha, A.data(), data());
        }
        else{
            gtensor<T, N, major> tmp(A);
            this->swap(tmp);
        }
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    template <typename E>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator+=(const tensor_expression<E> & A)
    {
//        std::cout <<"1" << std::endl;
        BOOST_STATIC_ASSERT(std::is_same<orientation_type, typename E::orientation_type>::value);
        if (size_ == 0) {
#ifdef GTENSOR_DEBUG
            std::cout << "original tensor is empty, A+=B will just act as A=B" << std::endl;
#endif
            self_type tmp(A);
            this->swap(tmp);
        }
        else{
            assert(dim_ == A().dim());
            for (size_t i = 0; i < size_; ++i) {
                data_[i] += A().operator[](i);
            }
        }
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator+=(const self_type & A)
    {
//        std::cout <<"2" << std::endl;
        if (size_==0) {
#ifdef GTENSOR_DEBUG
            std::cout << "original tensor is empty, A+=B will just act as A=B" << std::endl;
#endif
            self_type tmp(A);
            this->swap(tmp);
        }
        else{
            assert(dim_ == A.dim());
            data_ += A.data_;
        }
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    template <typename E>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator-=(const tensor_expression<E> & A)
    {
        //        std::cout <<"1" << std::endl;
        BOOST_STATIC_ASSERT(std::is_same<orientation_type, typename E::orientation_type>::value);
        if (size_ == 0) {
#ifdef GTENSOR_DEBUG
            std::cout << "original tensor is empty, A-=B will just act as A=-B" << std::endl;
#endif
            self_type tmp(A);
            tmp*=(-1);
            this->swap(tmp);
        }
        else{
            assert(dim_ == A().dim());
            for (size_t i = 0; i < size_; ++i) {
                data_[i] -= A().operator[](i);
            }
        }
        return *this;
    }

    
    template <typename T, size_t N, typename major>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator-=(const self_type & A)
    {
        if (size_==0) {
#ifdef GTENSOR_DEBUG
            std::cout << "original tensor is empty, A-=B will just act as A=-B" << std::endl;
#endif
            self_type tmp(A);
            tmp*=(-1);
            this->swap(tmp);
        }
        else{
            assert(dim_ == A.dim());
            data_ -= A.data_;
        }
        return *this;
    }
    
    
    template <typename T, size_t N, typename major>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator+=(value_type A)
    {
        data_ += A;
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator-=(value_type A)
    {
        data_ -= A;
        return *this;
    }

    
    template <typename T, size_t N, typename major>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator*=(value_type alpha)
    {
        data_*=alpha;
        return *this;
    }
    
    
    template <typename T, size_t N, typename major>
    const typename gtensor<T, N, major>::self_type
    &gtensor<T, N, major>::operator/=(value_type alpha)
    {
        data_/=alpha;
        return *this;
    }
    
    template <typename T, size_t N, typename major>
    template<typename T1>
    typename std::enable_if<std::is_scalar<T1>::value || boost::is_complex<T1>::value, const typename gtensor<T, N, major>::self_type>::type
    &gtensor<T, N, major>::rowscale(const diagmatrix<T1> & B)
    {
        BOOST_STATIC_ASSERT(std::is_same<major, label::column_major>::value);
        BOOST_STATIC_ASSERT(N>0);
        size_t size1 = dim_[0], size2 = 1, i;
        for (i = 1; i < N; ++i) {size2 *= dim_[i];}
        assert(B.size() == size1);
        major maj;
        tensor::general_rowscale(this->data(), size1, size2, B.data(), maj);
        return *this;

    }
    
    template <typename T, size_t N, typename major>
    template<typename T1>
    typename std::enable_if<std::is_scalar<T1>::value || boost::is_complex<T1>::value, const typename gtensor<T, N, major>::self_type>::type
    &gtensor<T, N, major>::colscale(const diagmatrix<T1> & B)
    {
        BOOST_STATIC_ASSERT(N>0);
        size_t size1 = 1, size2 = dim_[N-1], i;
        for (i = 0; i < N-1; ++i) {size1 *= dim_[i];}
        assert(B.size() == size2);
        major maj;
        tensor::general_colscale(this->data(), size1, size2, B.data(), maj);
        return *this;
    }


    
    
    template <typename T, size_t N, typename major>
    void gtensor<T, N, major>::swap(self_type & A)
    {
        size_type Asize = A.size_;
        dim_.swap(A.dim_);
        cudim_.swap(A.cudim_);
        data_.swap(A.data_);
        A.size_ = size_;
        size_ = Asize;
    }
    
    template <typename T, size_t N, typename major>
    void gtensor<T, N, major>::reshape(const dimvector_type & vec)
    {
        assert(size_ == tensor::prod<N>(vec));
        dim_ = vec;
        orientation_type maj;
        tensor::dim2cudim<N>(dim_, cudim_, maj);
    }
    

    
    template <typename T, size_t N, typename major>
    void gtensor<T, N, major>::resize(const dimvector_type & index, bool keep) //tested!
    {
        size_t size = tensor::detail::prod(index.data(), N);
        orientation_type maj;
        if (size == 0) {this->clear(); return;}
        if (size == size_) {
            this->reshape(index);
        }
        else{
            if (!keep) {
                dim_ = index;
                tensor::dim2cudim(dim_.data(), cudim_.data(), N, maj);
                size_ = size, data_.resize(size_);
            }
            else{
                dimvector_type minindex, cudim;
                tensor::detail::min(index.data(), dim_.data(), minindex.data(), N);
                tensor::dim2cudim(minindex.data(), cudim.data(), N, maj);
                size_t i, n = tensor::detail::prod(minindex.data(), N);
                self_type temp(index, 0);
                for (i = 0; i < n; ++i) {
                    
                    tensor::sind2mind(i, minindex.data(), cudim.data(), N, maj);
                    temp(minindex) = this->operator()(minindex);
                }
                this->swap(temp);
            }
        }
    }
    template <typename T, size_t N, typename major>
    void gtensor<T, N, major>::write()const
    {
        orientation_type ori;
        tensor::print_mult_array(this->data(), dim_.data(), ori, N);
    }
    
    
    /**********************************************************
     some nonmember functions.
     *********************************************************/
    template <typename T, size_t N, typename major>
    void swap(gtensor<T, N, major> & A, gtensor<T, N, major> & B){A.swap(B);}
    
    template <typename T1, typename T2, size_t N, typename major1, typename major2>
    inline void permute(const gtensor<T1, N, major1> & A, gtensor<T2, N, major2> & B, std::array<size_t, N> hashtable)
    {
        std::array<size_t, N> dimB;
        for (size_t i = 0; i < N; ++i) { hashtable[i] -= 1;}
        tensor::sortlist(A.dim(), dimB, hashtable);
        B.resize(dimB);
        tensor::permute_mult_array(A.data(), A.cudim(), A.size(), major2(), B.data(), B.cudim(), hashtable);
    }
    
    template <typename T, size_t N, typename major>
    inline gtensor<T, N, major> permute(const gtensor<T, N, major> & A, std::array<size_t, N> hashtable)
    {
        gtensor<T, N, major> B;
        permute(A, B, hashtable);
        return B;
    }
    
    template <typename T, size_t N, typename major>
    inline void circular_shift(const gtensor<T, N, major> & A, gtensor<T, N, major> & B, int shift, T sca=1.0)
    {
        std::array<size_t, N> dimB;
        tensor::detail::circular_shift_index(A.dim().data(), dimB.data(), N, shift);
        B.resize(dimB);
        major maj;
        label::transpose tran;
        tensor::circular_shift_mult_array(A.data(), A.dim().data(), maj, tran, A.size(), B.data(), sca, N, shift);

    }
    
    
    template <typename T, size_t N, typename major>
    inline gtensor<T, N, major> circular_shift(const gtensor<T, N, major> & A, int shift, T sca=1.0)
    {
        gtensor<T, N, major> B;
        circular_shift(A, B, shift, sca);
        return B;
    }
    
    
    
    /**********************************************************
     some friend functions.
     *********************************************************/
    template<size_t N2, typename T1, size_t N1, typename major>
    gtensor<T1, N2, major> reshape(gtensor<T1, N1, major> & A, std::array<size_t, N2> dim)
    {
        gtensor<T1, N2, major> B;
        B.dim_.swap(dim);
        major maj;
        tensor::dim2cudim<N2>(B.dim_, B.cudim_, maj);
        B.size_ = A.size_;
        B.data_.swap(A.data_);
        A.clear();
        return B;
    }


    
    
    template <typename T, size_t N>
    class gtensor<T, N, typename std::enable_if<N ==0, label::column_major>::type>
    : public tensor_expression<gtensor<T, N, typename std::enable_if<N ==0, label::column_major>::type> > {
        typedef gtensor<T, 0> self_type;
    public:
        typedef label::column_major orientation_type;
        typedef T   value_type;
        typedef size_t size_type;
        typedef typename traits::num_traits<value_type>::magnitude_type magnitude_type;
        typedef std::array<size_t, 0> dimvector_type;

        gtensor():data_(0){};
        gtensor(value_type A):data_(A){};
        gtensor(dimvector_type index):data_(0){}
        
        const self_type &operator=(value_type A){
            data_ = A;
            return *this;
        }
        
        value_type* data(){return &data_;};
        const value_type* data()const{return &data_;};
        size_type leadingdim()const{return 1;}
        size_type size1()const{return 1;};
        size_type size2()const{return 1;};
        
        void resize(dimvector_type index){};
        
        void write()const{
            std::cout <<"(" << data_ <<")"<< std::endl;
        }
        
    private:
        T data_;
    };
    

    
    
    
    // some print function
    template<typename E>
    void print(const tensor_expression<E> & A)
    {
        A().write();
    }

}


#endif /* gtensor_h */


