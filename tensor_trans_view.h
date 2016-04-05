//
//  tensor_trans_view.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_trans_view_h
#define guochu_gtensor_tensor_trans_view_h

#include "tensor_expression.h"
#include "tensor_label.h"


namespace guochu {
    namespace tensor{
        
        //only for column major matrix
        template<typename DT, typename TR, typename Maj>
        class matrix_trans_const_view:
        public tensor_expression<matrix_trans_const_view<DT, TR, Maj> >{
            typedef matrix_trans_const_view<DT, TR, Maj> self_type;
        public:
            typedef TR trans_type;
            typedef Maj orientation_type;
            typedef size_t size_type;
            typedef DT scalar_type;
            typedef DT value_type;
            typedef std::array<size_t, 2> dimvector_type;
            
            matrix_trans_const_view(const DT*data, size_t i, size_t j);
            
            
            size_t size1()const{return size1_;}
            size_t size2()const{return size2_;}
            size_t leadingdim()const{return leadingdim_;}
            size_t size()const{return size1_*size2_;}
            size_t rank()const{return 2;}
            inline dimvector_type dim()const{dimvector_type dim; dim[0] = size1_, dim[1] = size2_; return dim;}
            const scalar_type* data()const{return data_;}
            inline scalar_type operator()(size_t i, size_t j)const;
            inline scalar_type operator[](size_t i)const;
            void write()const;
            
        private:
            const scalar_type* data_;
            size_t size1_;
            size_t size2_;
            size_t leadingdim_;
        };
        
        template<typename DT, typename TR, typename Maj>
        matrix_trans_const_view<DT, TR, Maj>::matrix_trans_const_view(const DT*data, size_t i, size_t j): //tested!
        data_(data)
        {
            BOOST_STATIC_ASSERT(std::is_same<Maj, label::column_major>::value || std::is_same<Maj, label::row_major>::value);
            BOOST_STATIC_ASSERT(std::is_same<TR, label::no_transpose>::value || std::is_same<TR, label::transpose>::value
                                || std::is_same<TR, label::conjugate>::value);
            size1_ = std::is_same<TR, label::no_transpose>::value?i:j;
            size2_ = std::is_same<TR, label::no_transpose>::value?j:i;
            leadingdim_ = std::is_same<Maj, label::column_major>::value?i:j;
        }
        
        template<typename DT, typename TR, typename Maj>
        typename matrix_trans_const_view<DT, TR, Maj>::scalar_type
        matrix_trans_const_view<DT, TR, Maj>::operator()(size_t i, size_t j)const //tested!
        {
            if (std::is_same<Maj, label::column_major>::value){
                if (std::is_same<TR, label::no_transpose>::value) {
                    return data_[i + j*size1_];
                }
                else if (std::is_same<TR, label::transpose>::value)
                {
                    return data_[j + i*size2_];
                }
                else{
                    return conj(data_[j + i*size2_]);
                }
            }
            else{
                if (std::is_same<TR, label::no_transpose>::value) {
                    return data_[j + i*size2_];
                }
                else if (std::is_same<TR, label::transpose>::value){
                    return data_[i + j*size1_];
                }
                else{
                    return conj(data_[i + j*size1_]);
                }
            }
        }
        
        template<typename DT, typename TR, typename Maj>
        typename matrix_trans_const_view<DT, TR, Maj>::scalar_type
        matrix_trans_const_view<DT, TR, Maj>::operator[](size_t i)const
        {
            size_t a, b;
            if (std::is_same<Maj, label::column_major>::value){
                a = i%size1_;
                b = i/size1_;
            }
            else{
                a = i/size2_;
                b = i%size2_;
            }
            return this->operator()(a, b);
        }
        
        template<typename DT, typename TR, typename Maj>
        void matrix_trans_const_view<DT, TR, Maj>::write()const
        {
            std::cout << "(" << this->size1() <<","<< this->size2() << ")\n";
            size_t i, j;
            std::cout << "[ ";
            for (i=0; i<this->size1(); ++i) {
                std::cout << "[ ";
                for (j=0; j<this->size2(); ++j) {
                    std::cout << this->operator()(i,j) << ",";
                }
                std::cout << " ], " << std::endl;
            }
            std::cout<<" ]\n";
        }
        
        
        
        //nonmember functions
        
        
        
    }
}



#endif /* tensor_trans_view_h */
