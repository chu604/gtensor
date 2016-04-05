//
//  tensor_label.h
//  shifu_project_10
//
//  Created by guochu on 13/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

#ifndef guochu_gtensor_tensor_label_h
#define guochu_gtensor_tensor_label_h


#include <boost/mpl/char.hpp>
#include <boost/numeric/bindings/tag.hpp>

namespace guochu {
    
    namespace label{
        
        struct dense{};
        struct sparse{};
        struct identity{};
        using boost::numeric::bindings::tag::row_major;
        using boost::numeric::bindings::tag::column_major;
        using boost::numeric::bindings::tag::transpose;
        using boost::numeric::bindings::tag::no_transpose;
        using boost::numeric::bindings::tag::conjugate;
        using boost::numeric::bindings::tag::left;
        using boost::numeric::bindings::tag::right;
        
        
        template< typename T >
        struct tensor_option {};
        
        template<>
        struct tensor_option< transpose >: boost::mpl::char_< 'T' > {};
        
        template<>
        struct tensor_option< row_major >: boost::mpl::char_< 'R' > {};
        
        template<>
        struct tensor_option< column_major >: boost::mpl::char_< 'C' > {};
        
        template<>
        struct tensor_option< no_transpose >: boost::mpl::char_< 'N' > {};
        
        template<>
        struct tensor_option< conjugate >: boost::mpl::char_< 'C' > {};
        
    }
    
    
    
}


#endif /* tensor_label_h */
