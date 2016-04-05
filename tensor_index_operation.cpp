//
//  tensor_index_operation.cpp
//  shifu_project_11
//
//  Created by guochu on 12/3/16.
//  Copyright © 2016 guochu. All rights reserved.
//

//#include "tensor_index_operation.hpp"
#include <assert.h>
#include <complex>
#include <vector>

namespace guochu {
    
    
    double conj(double a){return a;}
    std::complex<double> conj(std::complex<double> a){return std::conj(a);}
    double real(double a){return a;}
    double real(std::complex<double> a){return std::real(a);}
    double imag(double a){return 0.;}
    double imag(std::complex<double> a){return std::imag(a);}
    
    
    
    
    namespace tensor {
        namespace detail {
            
#define gcsize_t size_t
            
            
            // basic operation for linear array, over double and complex double
            
            
            
            void conj(const double* a1, double* a2, gcsize_t n)
            {
                gcsize_t i = 0;
                while (i < n) {
                    *a2 = *a1;
                    ++a2, ++a1, ++i;
                }
            }
            
            void conj(const std::complex<double>* a1, std::complex<double>* a2, gcsize_t n)
            {
                gcsize_t i = 0;
                while (i < n) {
                    *a2 = conj(*a1);
                    ++a2, ++a1, ++i;
                }
            }
            
            void real(const double* a1, double* a2, gcsize_t n)
            {
                gcsize_t i = 0;
                while (i < n) {
                    *a2 = *a1;
                    ++a2, ++a1, ++i;
                }
            }
            
            void real(const std::complex<double>* a1, double* a2, gcsize_t n)
            {
                gcsize_t i = 0;
                while (i < n) {
                    *a2 = real(*a1);
                    ++a2, ++a1, ++i;
                }
            }
            
            void imag(const double* a1, double* a2, gcsize_t n)
            {
                gcsize_t i = 0;
                while (i < n) {
                    *a2 = 0;
                    ++i;
                }
            }
            
            void imag(const std::complex<double>* a1, double* a2, gcsize_t n)
            {
                gcsize_t i = 0;
                while (i < n) {
                    *a2 = imag(*a1);
                    ++a2, ++a1, ++i;
                }
            }
            
            
            void dim2cudim_col(const gcsize_t* dim, gcsize_t* cudim, gcsize_t N) //tested!
            {
                assert(N > 0);
                *cudim = 1;
                gcsize_t i=1, temp = 1;
                while (i++ < N) {
                    temp *= (*dim++);
                    *(++cudim) = temp;
                }
            }
            
            void dim2cudim_row(const gcsize_t* dim, gcsize_t* cudim, gcsize_t N) //tested!
            {
                assert(N > 0);
                gcsize_t temp = 1;
                dim += --N;
                cudim += N;
                *cudim = 1;
                while (N--) {
                    //                std::cout <<N<<' ';
                    temp *= (*dim--);
                    *(--cudim) = temp;
                }
            }
            
            gcsize_t mind2sind_col(const gcsize_t* multindex, const gcsize_t* cudim, gcsize_t N) //tested!
            {
                assert(N > 0);
                gcsize_t singleindex = *multindex, i = 0;
                while (++i < N) {
                    //                std::cout <<i<<' ';
                    singleindex += (*(++multindex))*(*(++cudim));
                }
                return singleindex;
            }
            
            gcsize_t mind2sind_row(const gcsize_t* multindex, const gcsize_t* cudim, gcsize_t N) //tested!
            {
                assert(N > 0);
                gcsize_t i = 0, singleindex = 0;
                --N;
                while (i++ < N) {
                    //                std::cout <<i<<' ';
                    singleindex += (*multindex++)*(*cudim++);
                }
                singleindex += *multindex;
                return singleindex;
            }
            
            gcsize_t mind2sind_col(const gcsize_t* multindex,
                                   const gcsize_t* startindex,
                                   const gcsize_t* cudim,
                                   gcsize_t N)
            {
                assert(N > 0);
                gcsize_t singleindex=*multindex++ + *startindex++, i = 0;
                while (++i < N) {
                    singleindex += (*multindex++ + *startindex++)*(*cudim++);
                }
                return singleindex;
            }
            
            gcsize_t mind2sind_row(const gcsize_t* multindex,
                                   const gcsize_t* startindex,
                                   const gcsize_t* cudim, gcsize_t N)
            {
                assert(N > 0);
                gcsize_t singleindex = 0, i = 0;
                --N;
                while (i++ < N) {
                    singleindex += (*multindex++ + *startindex++)*(*cudim++);
                }
                singleindex += (*multindex++ + *startindex++);
                return singleindex;
            }
            
            void sind2mind_col(gcsize_t singleindex, gcsize_t* multindex,
                               const gcsize_t* cudim, gcsize_t N)  //test!
            {
                assert(N > 0);
                multindex += --N;
                cudim += N;
                while (N--) {
                    *multindex-- = singleindex/(*cudim);
                    singleindex %= (*cudim--);
                }
                *multindex = singleindex/(*cudim);
            }
            
            void sind2mind_row(gcsize_t singleindex, gcsize_t* multindex,
                               const gcsize_t* cudim, gcsize_t N)  //test!
            {
                gcsize_t i = 0;
                while (++i < N) {
                    *multindex++ = singleindex/(*cudim);
                    singleindex %= (*cudim++);
                }
                *multindex = singleindex/(*cudim);
            }
            
            
            void sortlist(const gcsize_t* listin,
                          gcsize_t* listout, const gcsize_t* hash, gcsize_t N) //tested!
            {
                assert(N > 0);
                gcsize_t i = 0;
                *listout = listin[*hash];
                while (++i < N) {
                    *(++listout) = listin[*(++hash)];
                }
            }
            
            /***********************************************************************************
             this function is a bit hard to explain...
             for a multidimensional tensor with cudimout, we have a index out, and we have anoth-
             er multidimensioanal which has the same rank, and has the cumulate dimension cudimin
             . the two tensor indexes are related by hash, which is a permutation from 1 to rank.
             size is the rank of two tensors. ain is the out put, which is the map of aout. hash
             maps the index hash[i] into i.
             ***********************************************************************************/
            
            gcsize_t mapsind2sindbyhashcol(const gcsize_t* cudimin, gcsize_t aout,
                                           const gcsize_t* cudimout, const gcsize_t* hash, gcsize_t N) //tested!
            {
                assert(N > 0);
                gcsize_t ain = 0;
                cudimout += --N;
                hash += N;
                while (N--) {
                    ain += (aout/(*cudimout))*cudimin[*hash--];
                    aout %= (*cudimout--);
                }
                ain += (aout/(*cudimout))*cudimin[*hash];
                return ain;
            }
            
            gcsize_t mapsind2sindbyhashrow(const gcsize_t* cudimin, gcsize_t aout,
                                           const gcsize_t* cudimout, const gcsize_t* hash, gcsize_t N) //tested!
            {
                assert(N > 0);
                gcsize_t ain = 0, i = 0;
                while (++i < N) {
                    ain += (aout/(*cudimout))*(cudimin[*hash++]);
                    aout %= (*cudimout++);
                }
                ain += (aout/(*cudimout))*(cudimin[*hash]);
                return ain;
            }
            
            
            gcsize_t mapsind2sindcol2row(const gcsize_t* cudimrow, gcsize_t aout,
                                         const gcsize_t* cudimcol, gcsize_t N) //tested!
            {
                gcsize_t ain = 0;
                cudimcol += --N;
                cudimrow += N;
                while (N--) {
                    ain += (aout/(*cudimcol))*(*cudimrow--);
                    aout %= (*cudimcol--);
                }
                ain += (aout/(*cudimcol))*(*cudimrow);
                return ain;
            }
            
            gcsize_t mapsind2sindrow2col(const gcsize_t* cudimcol, gcsize_t aout,
                                         const gcsize_t* cudimrow, gcsize_t N) //tested!
            {
                gcsize_t ain = 0, i = 0;
                while (++i < N) {
                    ain += (aout/(*cudimrow))*(*cudimcol++);
                    aout %= (*cudimrow++);
                }
                ain += (aout/(*cudimrow))*(*cudimcol);
                return ain;
            }
            
            gcsize_t sindview2sind_col(const gcsize_t* cudimall, gcsize_t asubview,
                                       const gcsize_t* cudimasub,
                                       const gcsize_t* startindex, gcsize_t N) //tested!
            {
                gcsize_t a = 0;
                cudimasub += --N;
                startindex += N;
                cudimall += N;
                while (N--) {
                    a += (asubview/(*cudimasub)+(*startindex--))*(*cudimall--);
                    asubview %= (*cudimasub--);
                }
                a += (asubview/(*cudimasub)+(*startindex))*(*cudimall);
                return a;
            }
            
            gcsize_t sindview2sind_row(const gcsize_t* cudimall, gcsize_t asubview,
                                       const gcsize_t* cudimasub,
                                       const gcsize_t* startindex, gcsize_t N) //tested!
            {
                gcsize_t a = 0, i = 0;
                while (++i < N) {
                    a += (asubview/(*cudimasub)+(*startindex++))*(*cudimall++);
                    asubview %= (*cudimasub++);
                }
                a += (asubview/(*cudimasub)+(*startindex))*(*cudimall);
                return a;
            }
            
            gcsize_t sindview2sind_col(const gcsize_t* cudima, gcsize_t asub,
                                       const gcsize_t* cudimasub, gcsize_t N)
            {
                gcsize_t a = 0;
                cudimasub += --N;
                cudima += N;
                while (N--) {
                    a += (asub/(*cudimasub))*(*cudima--);
                    asub %= (*cudimasub--);
                }
                a += (asub/(*cudimasub))*(*cudima);
                return a;
            }
            
            gcsize_t sindview2sind_row(const gcsize_t* cudima, gcsize_t asub,
                                       const gcsize_t* cudimasub, gcsize_t N)
            {
                gcsize_t a = 0, i = 0;
                while (++i < N) {
                    a += (asub/(*cudimasub))*(*cudima++);
                    asub %= (*cudimasub++);
                }
                a += (asub/(*cudimasub))*(*cudima);
                return a;
            }
            
            
            double mean(const double* vec, gcsize_t size)
            {
                double meanvalue = 0.0;
                for (size_t i = 0; i<size; ++i) {
                    meanvalue += vec[i];
                }
                meanvalue/=size;
                return meanvalue;
            }
            
            double stdvar(const double* vec, gcsize_t size)
            {
                double meanvalue = mean(vec, size), value=0.0;
                for (size_t i=0; i < size; ++i) {
                    value+=(vec[i]-meanvalue)*(vec[i]-meanvalue);
                }
                return value;
            }
            
            
            gcsize_t mind2sind_col(gcsize_t i, gcsize_t j, const gcsize_t* cudim)
            {
                return i + j*cudim[1];
            }
            
            gcsize_t mind2sind_col(gcsize_t i, gcsize_t j, gcsize_t k, const gcsize_t* cudim)
            {
                return i + j*cudim[1] + k*cudim[2];
            }
            
            gcsize_t mind2sind_col(gcsize_t i, gcsize_t j, gcsize_t k, gcsize_t l, const gcsize_t* cudim)
            {
                return i + j*cudim[1] + k*cudim[2] + l*cudim[3];
            }
            
            gcsize_t mind2sind_col(gcsize_t i, gcsize_t j, gcsize_t k, gcsize_t l, gcsize_t m, const gcsize_t* cudim)
            {
                return i + j*cudim[1] + k*cudim[2] + l*cudim[3] + m*cudim[4];
            }
            
            gcsize_t mind2sind_col(gcsize_t i, gcsize_t j, gcsize_t k, gcsize_t l, gcsize_t m, gcsize_t n, const gcsize_t* cudim)
            {
                return i + j*cudim[1] + k*cudim[2] + l*cudim[3] + m*cudim[4] + n*cudim[5];
            }
            
            gcsize_t mind2sind_row(gcsize_t i, gcsize_t j, const gcsize_t* cudim)
            {
                return i*cudim[0] + j;
            }
            
            gcsize_t mind2sind_row(gcsize_t i, gcsize_t j, gcsize_t k, const gcsize_t* cudim)
            {
                return i*cudim[0] + j*cudim[1] + k;
            }
            
            gcsize_t mind2sind_row(gcsize_t i, gcsize_t j, gcsize_t k, gcsize_t l, const gcsize_t* cudim)
            {
                return i*cudim[0] + j*cudim[1] + k*cudim[2] + l;
            }
            
            gcsize_t mind2sind_row(gcsize_t i, gcsize_t j, gcsize_t k, gcsize_t l, gcsize_t m, const gcsize_t* cudim)
            {
                return i*cudim[0] + j*cudim[1] + k*cudim[2] + l*cudim[3] + m;
            }
            
            gcsize_t mind2sind_row(gcsize_t i, gcsize_t j, gcsize_t k, gcsize_t l, gcsize_t m, gcsize_t n, const gcsize_t* cudim)
            {
                return i*cudim[0] + j*cudim[1] + k*cudim[2] + l*cudim[3] + m*cudim[4] + n;
            }
            
            void circular_shift_index(const gcsize_t* Aold,
                                      gcsize_t* Anew, gcsize_t N, int shift)
            {
                gcsize_t rest, i;
                if (shift >= 0) {
                    shift %= N;
                    rest = N-shift;
                    for (i = 0; i < shift; ++i) {
                        Anew[rest + i] = Aold[i];
                    }
                    for (i = shift; i < N; ++i) {
                        Anew[i - shift] = Aold[i];
                    }
                }
                else{
                    shift = -shift;
                    shift %= N;
                    rest = N-shift;
                    for (i = 0; i < rest; ++i) {
                        Anew[i + shift] = Aold[i];
                    }
                    for (i = rest; i < N; ++i) {
                        Anew[i - rest] = Aold[i];
                    }
                }
            }
            
            
            
#undef gcsize_t
            
        }
    }
}
