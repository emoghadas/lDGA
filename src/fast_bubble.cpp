#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using namespace nb::literals;

#include <cmath>
#include <complex>



// tight binding dispersion relation for 3 dimensions
double ek_3d(const std::vector<double>& k, double t=1.){
    double epsilon =  -2. * t * (cos(k[0]) + cos(k[1]) + cos(k[2]));
    return epsilon;
}


// q dependent bubble with k-integration
void calc_bubble(
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& chi0,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& sigma,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& sigma_w,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& iv,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& ivw,
    double mu,
    double beta,
    const std::vector<double>& q,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>& kgrid){

        // init Gk and Gkq
        std::complex<double> Gk;
        std::complex<double> Gkq;

        // loop through matsubaras and kgrid
        auto x = chi0.view();
        for(size_t nu_i=0; nu_i<x.shape(0); nu_i++){
            std::complex<double> sum (0., 0.);
            for (size_t xi=0; xi<kgrid.shape(0); xi++){
                for (size_t yi=0; yi<kgrid.shape(0); yi++){
                    for (size_t zi=0; zi<kgrid.shape(0); zi++){
                        std::vector<double> kvec = {kgrid(xi), kgrid(yi), kgrid(zi)};
                        std::vector<double> kqvec = {kgrid(xi)+q[0], kgrid(yi)+q[1], kgrid(zi)+q[2]};

                        Gk = 1./(iv(nu_i) + mu - ek_3d(kvec) - sigma(nu_i));
                        Gkq = 1./(ivw(nu_i) + mu - ek_3d(kqvec) - sigma_w(nu_i));
                        sum -= beta * Gk * Gkq/pow(kgrid.shape(0), 3);
                    }
                }
            }
            chi0(nu_i) = sum;
        }
    }


// q dependent bubble with k-integration
void calc_bubble_gl(
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& chi0,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& sigma,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& sigma_w,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& iv,
    nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>& iv_w,
    double mu,
    double beta,
    const std::vector<double>& q,
    int kpoints){

        // gauss legendre quadrature points to order n=5
        int Ng = 5;
        std::vector<double> tstep = {-0.9061798459386640, -0.53846931010568309, 0.0000000000000000, 0.53846931010568309, 0.9061798459386640};
        std::vector<double> wg = {0.2369268850561891, 0.47862867049936650, 0.5688888888888889, 0.47862867049936650, 0.2369268850561891};

        // initialize kgrid
        std::vector<std::vector<double>> kgrid(kpoints, std::vector<double> (Ng, 0.));
        for(int ki=0; ki<kpoints; ki++){
            for(int i=0; i<Ng; i++){
                double k = (M_PI/(1.0*kpoints)) * (tstep[i] + 2.0*ki + 1.0) - M_PI;
                kgrid[ki][i] = k;
            }
        }

        // init Gk and Gkq
        std::complex<double> Gk;
        std::complex<double> Gkq;

        // loop through matsubaras
        auto x = chi0.view();
        for(size_t nu_i=0; nu_i<x.shape(0); nu_i++){
            // loop through GL quadrature points
            std::complex<double> sum (0., 0.);
            for(int i=0; i<Ng; i++){
                for(int j=0; j<Ng; j++){
                    for(int l=0; l<Ng; l++){
                        // now loop through kpoints
                        for (size_t xi=0; xi<kpoints; xi++){
                            for (size_t yi=0; yi<kpoints; yi++){
                                for (size_t zi=0; zi<kpoints; zi++){
                                    std::vector<double> kvec = {kgrid[xi][i], kgrid[yi][j], kgrid[zi][l]};
                                    std::vector<double> kqvec = {kgrid[xi][i]+q[0], kgrid[yi][j]+q[1], kgrid[zi][l]+q[2]};

                                    Gk = 1./(iv(nu_i) + mu - ek_3d(kvec) - sigma(nu_i));
                                    Gkq = 1./(iv_w(nu_i) + mu - ek_3d(kqvec) - sigma_w(nu_i));
                                    sum -= beta * wg[i] * wg[j] * wg[l] * Gk * Gkq/pow(2.0*kpoints, 3);
                                }
                            }
                        }
                    }
                }
            }
            
            chi0(nu_i) = sum;
        }
    } 


NB_MODULE(_fast_bubble, m) {
    m.def("ek_3d", &ek_3d, "k"_a, "t"_a=1);
    m.def("calc_bubble", &calc_bubble);
    m.def("calc_bubble_gl", &calc_bubble_gl);
}