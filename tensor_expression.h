//
//  tensor_expression.h
//  gtensor
//
//  Created by guochu on 20/11/15.
//  Copyright © 2015 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_expression_h
#define guochu_gtensor_tensor_expression_h



namespace guochu {
 
    template<typename E>
    class tensor_expression{
    private:
        typedef tensor_expression<E> self_type;
    public:
        typedef E expression_type;
        
        const expression_type &operator() ()const{
           return *static_cast<const expression_type *> (this);
        }
        expression_type &operator() (){
            return *static_cast<expression_type *> (this);
        }
    };
    
    

    
    
}

#endif /* tensor_expression_h */
