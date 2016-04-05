//
//  tensor_block_view.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_block_view_h
#define guochu_gtensor_tensor_block_view_h

#include "tensor_expression.h"

namespace guochu {
    namespace tensor{
        
        template<typename T, size_t N, typename Maj>
        class tensor_block_view
        :public tensor_expression<tensor_block_view<T, N, Maj> >
        {
            typedef tensor_block_view<T, N, Maj> self_type;
        public:
            typedef std::array<size_t, N> dimvector_type;
            typedef Maj orientation_type;
            typedef size_t size_type;
            typedef T scalar_type;
            typedef T value_type;
            
            tensor_block_view();
            
            tensor_block_view(T* data, const dimvector_type& superdim, const dimvector_type& supercudim,
                              const dimvector_type& startind, const dimvector_type& endind);
            
            
            inline scalar_type& operator[](size_type i);
            
            inline const scalar_type& operator[](size_type i)const;
            
            inline scalar_type &operator()(const dimvector_type & vec);
            
            const inline scalar_type &operator()(const dimvector_type & vec)const;
            
            
            
            template <typename M>
            const self_type &operator=(const tensor_expression<M> & A);
            
            const self_type &operator=(scalar_type A);
            
            template <typename M>
            const self_type& reduced_assign(const tensor_expression<M> & A);
            
            
            template<typename M>
            const self_type &operator+=(const tensor_expression<M> & A);
            template<typename M>
            const self_type &operator-=(const tensor_expression<M> & A);
            const self_type &operator+=(scalar_type A);
            const self_type &operator-=(scalar_type A);
            const self_type &operator*=(scalar_type alpha);
            const self_type &operator/=(scalar_type alpha);
            
            size_t size()const;
            
            void write()const;
            
            dimvector_type dim()const{return subdim_;}
            dimvector_type cudim()const{return subcudim_;}
            size_type leadingdim()const
            {
                BOOST_STATIC_ASSERT(N > 0);
                return std::is_same<orientation_type, label::column_major>::value?superdim_[0]:superdim_[N-1];
            };
            size_type size1()const{BOOST_STATIC_ASSERT(N > 0); return subdim_[0];};
            size_type size2()const{BOOST_STATIC_ASSERT(N > 1); return subdim_[1];};
            size_type size3()const{BOOST_STATIC_ASSERT(N > 2); return subdim_[2];};
            size_type size4()const{BOOST_STATIC_ASSERT(N > 3); return subdim_[3];};
            size_type size5()const{BOOST_STATIC_ASSERT(N > 4); return subdim_[4];};
            size_type size6()const{BOOST_STATIC_ASSERT(N > 5); return subdim_[5];};
            size_type size7()const{BOOST_STATIC_ASSERT(N > 6); return subdim_[6];};
            size_type size8()const{BOOST_STATIC_ASSERT(N > 7); return subdim_[7];};
            size_type sizeE()const{BOOST_STATIC_ASSERT(N > 0); return subdim_[N-1];}
            
            
        private:
            T* data_;
            const dimvector_type& superdim_;
            const dimvector_type& supercudim_;
            dimvector_type subdim_;
            dimvector_type subcudim_;
            dimvector_type startind_;
        };
        
        template<typename T, size_t N, typename Maj>
        tensor_block_view<T, N, Maj>::tensor_block_view()
        {
            startind_.fill(0), subdim_.fill(0), subcudim_.fill(0);
        }
        
        template<typename T, size_t N, typename Maj>
        tensor_block_view<T, N, Maj>::tensor_block_view(T* data, const dimvector_type& superdim, const dimvector_type& supercudim,
                                                        const dimvector_type& startind, const dimvector_type& endind):
        data_(data),
        superdim_(superdim),
        supercudim_(supercudim),
        startind_(startind)
        {
            orientation_type maj;
            detail::sub(endind.data(), startind_.data(), subdim_.data(), N);
            dim2cudim(subdim_.data(), subcudim_.data(), N, maj);
        }
        
        template<typename T, size_t N, typename Maj>
        typename tensor_block_view<T, N, Maj>::scalar_type&
        tensor_block_view<T, N, Maj>::operator[](size_type i)
        {
            orientation_type maj;
            size_t mapi = sindview2sind(supercudim_.data(), i, subcudim_.data(), startind_.data(), N, maj);
            assert(i <= size() && mapi < tensor::detail::prod(superdim_.data(), N));
            return data_[mapi];
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::scalar_type&
        tensor_block_view<T, N, Maj>::operator[](size_type i)const
        {
            orientation_type maj;
            size_t mapi = sindview2sind(supercudim_.data(), i, subcudim_.data(), startind_.data(), N, maj);
            assert(i <= size() && mapi < tensor::detail::prod(superdim_.data(), N));
            return data_[mapi];
        }
        
        template<typename T, size_t N, typename Maj>
        typename tensor_block_view<T, N, Maj>::scalar_type&
        tensor_block_view<T, N, Maj>::operator()(const dimvector_type & vec)
        {
#ifdef GTENSOR_DEBUG
            for (size_t i = 0; i < vec.size(); ++i) {
                if (vec[i] > subdim_[i]) {
                    std::stringstream ss("index ");
                    ss << i << " is out of range(" <<subdim_[i]<<").\n";
                    throw std::runtime_error(ss.str());
                }
            }
#endif
            Maj maj;
            size_t singleindex = mind2sind(vec.data(), startind_.data(), supercudim_.data(), N, maj);
            return data_->operator[](singleindex);
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::scalar_type&
        tensor_block_view<T, N, Maj>::operator()(const dimvector_type & vec)const
        {
#ifdef GTENSOR_DEBUG
            for (size_t i = 0; i < vec.size(); ++i) {
                if (vec[i] > subdim_[i]) {
                    std::stringstream ss("index ");
                    ss << i << " is out of range(" <<subdim_[i]<<").\n";
                    throw std::runtime_error(ss.str());
                }
            }
#endif
            Maj maj;
            size_t singleindex = mind2sind(vec.data(), startind_.data(), supercudim_.data(), N, maj);
            return data_->operator[](singleindex);
        }
        
        
        template<typename T, size_t N, typename Maj>
        template <typename M>
        const typename tensor_block_view<T, N, Maj>::self_type &
        tensor_block_view<T, N, Maj>::operator=(const tensor_expression<M> & A)
        {
            BOOST_STATIC_ASSERT(std::is_same<orientation_type, typename M::orientation_type>::value);
            assert(subdim_ == A().dim());
            for (size_t i=0; i<size(); ++i) {
                this->operator[](i) = A().operator[](i);
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        template <typename M>
        const typename tensor_block_view<T, N, Maj>::self_type&
        tensor_block_view<T, N, Maj>::reduced_assign(const tensor_expression<M> & A)
        {
            assert(reducable(subdim_, A().dim()));
            for (size_t i=0; i<size(); ++i) {
                this->operator[](i) = A().operator[](i);
            }
            return *this;
        }
        
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::self_type&
        tensor_block_view<T, N, Maj>::operator=(scalar_type A)
        {
            for (size_t i = 0; i < size(); ++i) {
                this->operator[](i) = A;
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        template<typename M>
        const typename tensor_block_view<T, N, Maj>::self_type&
        tensor_block_view<T, N, Maj>::operator+=(const tensor_expression<M> & A)
        {
            BOOST_STATIC_ASSERT(std::is_same<orientation_type, typename T::orientation_type>::value);
            assert(subdim_ == A().dim());
            for (size_t i = 0; i < A().size(); ++i) {
                this->operator[](i) += A().operator[](i);
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        template<typename M>
        const typename tensor_block_view<T, N, Maj>::self_type
        &tensor_block_view<T, N, Maj>::operator-=(const tensor_expression<M> & A)
        {
            BOOST_STATIC_ASSERT(std::is_same<orientation_type, typename T::orientation_type>::value);
            assert(subdim_ == A().dim());
            for (size_t i = 0; i < A().size(); ++i) {
                this->operator[](i) -= A().operator[](i);
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::self_type&
        tensor_block_view<T, N, Maj>::operator+=(scalar_type A)
        {
            for (size_t i = 0; i < size(); ++i) {
                this->operator[](i) += A;
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::self_type&
        tensor_block_view<T, N, Maj>::operator-=(scalar_type A)
        {
            for (size_t i = 0; i < size(); ++i) {
                this->operator[](i) -= A;
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::self_type&
        tensor_block_view<T, N, Maj>::operator*=(scalar_type A)
        {
            for (size_t i = 0; i < size(); ++i) {
                this->operator[](i) *= A;
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_view<T, N, Maj>::self_type
        &tensor_block_view<T, N, Maj>::operator/=(scalar_type A)
        {
            for (size_t i = 0; i < size(); ++i) {
                this->operator[](i) /= A;
            }
            return *this;
        }
        
        template<typename T, size_t N, typename Maj>
        size_t tensor_block_view<T, N, Maj>::size()const
        {
            return detail::prod(subdim_.data(), N);
        }
        
        template<typename T, size_t N, typename Maj>
        void tensor_block_view<T, N, Maj>::write()const
        {
            std::vector<scalar_type> tmp(size());
            for (size_t i = 0; i < size(); ++i) {
                tmp[i] = this->operator[](i);
            }
            Maj maj;
            return print_mult_array(tmp.data(), subdim_.data(), maj, N);
        }
        
        
        
        //constant view
        
        template<typename T, size_t N, typename Maj>
        class tensor_block_const_view:
        public tensor_expression<tensor_block_const_view<T, N, Maj> > {
            typedef tensor_block_const_view<T, N, Maj> self_type;
        public:
            typedef Maj orientation_type;
            typedef size_t size_type;
            typedef T scalar_type;
            typedef T value_type;
            typedef std::array<size_t, N>                dimvector_type;
            
            
            tensor_block_const_view();
            
            tensor_block_const_view(const T* data, const dimvector_type& superdim, const dimvector_type& supercudim,
                                    const dimvector_type& startind, const dimvector_type& endind);
            
            
            inline const scalar_type& operator[](size_type size1)const;
            
            inline const scalar_type &operator()(const dimvector_type & vec)const;
            
            
            bool empty()const{ return size() == 0;}
            
            size_type size()const;
            
            dimvector_type dim()const{return subdim_;}
            dimvector_type cudim()const{return subcudim_;}
            size_type leadingdim()const{BOOST_STATIC_ASSERT(N==2);return data_->size1();};
            size_type size1()const{BOOST_STATIC_ASSERT(N > 0); return subdim_[0];};
            size_type size2()const{BOOST_STATIC_ASSERT(N > 1); return subdim_[1];};
            size_type size3()const{BOOST_STATIC_ASSERT(N > 2); return subdim_[2];};
            size_type size4()const{BOOST_STATIC_ASSERT(N > 3); return subdim_[3];};
            size_type size5()const{BOOST_STATIC_ASSERT(N > 4); return subdim_[4];};
            size_type size6()const{BOOST_STATIC_ASSERT(N > 5); return subdim_[5];};
            size_type size7()const{BOOST_STATIC_ASSERT(N > 6); return subdim_[6];};
            size_type size8()const{BOOST_STATIC_ASSERT(N > 7); return subdim_[7];};
            size_type sizeE()const{BOOST_STATIC_ASSERT(N > 0); return subdim_[N-1];}
            
            void write()const;
            
        private:
            const T* data_;
            const dimvector_type& superdim_;
            const dimvector_type& supercudim_;
            dimvector_type subdim_;
            dimvector_type subcudim_;
            dimvector_type startind_;
        };
        
        template<typename T, size_t N, typename Maj>
        tensor_block_const_view<T, N, Maj>::tensor_block_const_view(const T* data, const dimvector_type& superdim, const dimvector_type& supercudim,
                                                                    const dimvector_type& startind, const dimvector_type& endind):
        data_(data),
        superdim_(superdim),
        supercudim_(supercudim),
        startind_(startind)
        {
            orientation_type maj;
            detail::sub(endind.data(), startind_.data(), subdim_.data(), N);
            dim2cudim(subdim_.data(), subcudim_.data(), N, maj);
        }
        
        
        
        template<typename T, size_t N, typename Maj>
        size_t tensor_block_const_view<T, N, Maj>::size()const
        {
            return detail::prod(subdim_.data(), N);
        }
        
        template<typename T, size_t N, typename Maj>
        void tensor_block_const_view<T, N, Maj>::write()const
        {
            std::vector<scalar_type> tmp(size());
            for (size_t i = 0; i < size(); ++i) {
                tmp[i] = this->operator[](i);
            }
            Maj maj;
            return print_mult_array(tmp.data(), subdim_.data(), maj, N);
        }
        
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_const_view<T, N, Maj>::scalar_type&
        tensor_block_const_view<T, N, Maj>::operator[](size_type i)const
        {
            orientation_type maj;
            size_t mapi = sindview2sind(supercudim_.data(), i, subcudim_.data(), startind_.data(), N, maj);
            assert(i <= size() && mapi < tensor::detail::prod(superdim_.data(), N));
            return data_[mapi];
        }
        
        template<typename T, size_t N, typename Maj>
        const typename tensor_block_const_view<T, N, Maj>::scalar_type&
        tensor_block_const_view<T, N, Maj>::operator()(const dimvector_type & vec)const
        {
#ifdef GTENSOR_DEBUG
            for (size_t i = 0; i < vec.size(); ++i) {
                assert(vec[i] < subdim_[i]);
            }
#endif
            Maj maj;
            size_t singleindex = mind2sind(vec.data(), startind_.data(), supercudim_.data(), N, maj);
            return data_[singleindex];
        }
        
        

    }
}


#endif /* tensor_block_view_h */
