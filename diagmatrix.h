//
//  diagmatrix.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_diagmatrix_h
#define guochu_gtensor_diagmatrix_h

#include <vector>
#include <boost/serialization/base_object.hpp>

namespace guochu {
    
    
    template<typename T>
    class diagmatrix:
    public std::vector<T>{
        typedef diagmatrix<T> self_type;
        typedef std::vector<T> base_type;
    public:
        typedef T value_type;
        diagmatrix():base_type(){};
        explicit diagmatrix(size_t L):base_type(L){};
        explicit diagmatrix(size_t L, value_type num):base_type(L, num){};
        diagmatrix(const base_type & A) : base_type(A){};
        diagmatrix(base_type&& A) : base_type(A){};
        void write()const;
        
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    
    template<typename T>
    diagmatrix<T> kron(const diagmatrix<T> & v1, const diagmatrix<T> & v2)
    {
        size_t size1 = v1.size(), size2 = v2.size(), size = size1*size2, i, j;
        diagmatrix<T> v(size);
        for (i = 0; i < size1; ++i) {
            for (j = 0; j < size2; ++j) {
                v[i*size2+j] = v1[i]*v2[j];
            }
        }
        return v;
    }
    
    template<typename T>
    void diagmatrix<T>::write()const
    {
        std::cout << "[ ";
        for (size_t i = 0; i < this->size(); i++) {
            std::cout << this->operator[](i) <<",";
        }
        std::cout <<" ]"<< std::endl;
    }
    
    template<typename T>
    void print(const diagmatrix<T> & mat)
    {
        mat.write();
    }

}


#endif /* diagmatrix_h */
