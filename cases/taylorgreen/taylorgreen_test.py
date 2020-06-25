import sys
import os
import copy
import shutil
import struct
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
from numpy import *

sys.path.append('../../python/')
import microhh_tools as mht


no_opts = {}

opt_mpi = {}

dict_resolution = {
    'itot016': {'grid': {'itot':  16, 'ktot':   8}},
    'itot032': {'grid': {'itot':  32, 'ktot':  16}},
    'itot064': {'grid': {'itot':  64, 'ktot':  32}},
    'itot128': {'grid': {'itot': 128, 'ktot':  64}},
    'itot256': {'grid': {'itot': 256, 'ktot': 128}}}

dict_order = {
    'swadvec2' : {'grid': {'swspatialorder': 2}, 'advec': {'swadvec': '2' }},
    'swadvec4' : {'grid': {'swspatialorder': 4}, 'advec': {'swadvec': '4' }},
    'swadvec4m': {'grid': {'swspatialorder': 4}, 'advec': {'swadvec': '4m'}}}


class Microhh:
    def __init__(self, iter, itot, ktot, loadtype, path):
        nx = itot
        ny = 1
        nz = ktot

        n = nx * nz
        # Set the correct string for the endianness
        en = '<'

        # Set the correct string for the loadtype
        if (loadtype == 'dp'):
            TF = 8
            ra = 'd'
        elif (loadtype == 'sp'):
            TF = 4
            ra = 'f'
        else:
            raise RuntimeError("The savetype has to be sp or dp")

        fstring = '{0}{1}' + ra

        # Read grid properties from grid.0000000
        n = nx * ny * nz
        fin = open("{0:s}/grid.{1:07d}".format(path, 0), "rb")
        raw = fin.read(nx * TF)
        self.x = array(struct.unpack(fstring.format(en, nx), raw))
        raw = fin.read(nx * TF)
        self.xh = array(struct.unpack(fstring.format(en, nx), raw))
        raw = fin.read(ny * TF)
        self.y = array(struct.unpack(fstring.format(en, ny), raw))
        raw = fin.read(ny * TF)
        self.yh = array(struct.unpack(fstring.format(en, ny), raw))
        raw = fin.read(nz * TF)
        self.z = array(struct.unpack(fstring.format(en, nz), raw))
        raw = fin.read(nz * TF)
        self.zh = array(struct.unpack(fstring.format(en, nz), raw))
        fin.close()

        fin = open("{0:s}/u.xz.00000.{1:07d}".format(path, iter), "rb")
        raw = fin.read(n * TF)
        tmp = array(struct.unpack(fstring.format(en, n), raw))
        del(raw)
        self.u = tmp.reshape((nz, ny, nx))
        del(tmp)
        fin.close()

        fin = open("{0:s}/w.xz.00000.{1:07d}".format(path, iter), "rb")
        raw = fin.read(n * TF)
        tmp = array(struct.unpack(fstring.format(en, n), raw))
        del(raw)
        self.w = tmp.reshape((nz, ny, nx))
        del(tmp)
        fin.close()

        fin = open("{0:s}/p.xz.00000.{1:07d}".format(path, iter), "rb")
        raw = fin.read(n * TF)
        tmp = array(struct.unpack(fstring.format(en, n), raw))
        del(raw)
        self.p = tmp.reshape((nz, ny, nx))
        del(tmp)
        fin.close()


class Get_ref:
    def __init__(self, x, xh, z, zh, visc, time):
        self.u = zeros((zh.size, 1, x .size))
        self.w = zeros((z .size, 1, xh.size))
        self.p = zeros((z .size, 1, x .size))

        for k in range(z.size):
            self.u[k, 0, :] = sin(2. * pi * xh) * cos(2. * pi * z[k]) * \
                exp(-8. * pi**2. * visc * time)
            self.w[k, 0, :] = -cos(2. * pi * x) * \
                sin(2. * pi * zh[k]) * exp(-8. * pi**2. * visc * time)
            self.p[k, 0, :] = (
                0.25 * (cos(4. * pi * x) + cos(4. * pi * z[k])) - 0.25) * (exp(-8. * pi**2. * visc * time)**2.)


class Get_error:
    def __init__(self, data, ref):
        dx = 1. / data.x.size
        dz = 0.5 / data.z.size

        self.u = 0.
        self.w = 0.
        self.p = 0.

        for k in range(data.z.size):
            self.u = self.u + sum(dx * dz * abs(data.u[k, :] - ref.u[k, :]))
            self.w = self.w + sum(dx * dz * abs(data.w[k, :] - ref.w[k, :]))
            self.p = self.p + sum(dx * dz * abs(data.p[k, :] - ref.p[k, :]))


def plot(case_name, case_dir, experiment, float_type):

    # Log the stream output to a file.
    sys.stdout = open('{}/{}_{}.log'.format(case_dir, case_name, experiment), 'w')

    t = 1
    time = 1.
    visc = (8. * pi**2. * 100.)**(-1.)

    ns = array([16, 32, 64, 128, 256])
    dxs = 1. / ns

    # 2nd order data
    data16_2nd = Microhh(
        t,
        16,
        8,
        float_type,
        '{}/{}_itot016_swadvec2'.format(case_dir, experiment))
    data32_2nd = Microhh(
        t,
        32,
        16,
        float_type,
        '{}/{}_itot032_swadvec2'.format(case_dir, experiment))
    data64_2nd = Microhh(
        t,
        64,
        32,
        float_type,
        '{}/{}_itot064_swadvec2'.format(case_dir, experiment))
    data128_2nd = Microhh(
        t,
        128,
        64,
        float_type,
        '{}/{}_itot128_swadvec2'.format(case_dir, experiment))
    data256_2nd = Microhh(
        t,
        256,
        128,
        float_type,
        '{}/{}_itot256_swadvec2'.format(case_dir, experiment))

    ref16_2nd = Get_ref(data16_2nd .x, data16_2nd .xh,
                       data16_2nd .z, data16_2nd .zh, visc, time)
    ref32_2nd = Get_ref(data32_2nd .x, data32_2nd .xh,
                       data32_2nd .z, data32_2nd .zh, visc, time)
    ref64_2nd = Get_ref(data64_2nd .x, data64_2nd .xh,
                       data64_2nd .z, data64_2nd .zh, visc, time)
    ref128_2nd = Get_ref(data128_2nd.x, data128_2nd.xh,
                        data128_2nd.z, data128_2nd.zh, visc, time)
    ref256_2nd = Get_ref(data256_2nd.x, data256_2nd.xh,
                        data256_2nd.z, data256_2nd.zh, visc, time)

    err16_2nd = Get_error(data16_2nd, ref16_2nd)
    err32_2nd = Get_error(data32_2nd, ref32_2nd)
    err64_2nd = Get_error(data64_2nd, ref64_2nd)
    err128_2nd = Get_error(data128_2nd, ref128_2nd)
    err256_2nd = Get_error(data256_2nd, ref256_2nd)

    errsu_2nd = array(
        [err16_2nd.u, err32_2nd.u, err64_2nd.u, err128_2nd.u, err256_2nd.u])
    errsw_2nd = array(
        [err16_2nd.w, err32_2nd.w, err64_2nd.w, err128_2nd.w, err256_2nd.w])
    errsp_2nd = array(
        [err16_2nd.p, err32_2nd.p, err64_2nd.p, err128_2nd.p, err256_2nd.p])

    print('errors p_2nd', errsp_2nd)
    if(t > 0):
        print('convergence u_2nd', (log(
            errsu_2nd[-1]) - log(errsu_2nd[0])) / (log(dxs[-1]) - log(dxs[0])))
        print('convergence w_2nd', (log(
            errsw_2nd[-1]) - log(errsw_2nd[0])) / (log(dxs[-1]) - log(dxs[0])))
    print('convergence p_2nd',
          (log(errsp_2nd[-1]) - log(errsp_2nd[0])) / (log(dxs[-1]) - log(dxs[0])))

    # 42 order data
    data16_4m = Microhh(
        t,
        16,
        8,
        float_type,
        '{}/{}_itot016_swadvec4m'.format(case_dir, experiment))
    data32_4m = Microhh(
        t,
        32,
        16,
        float_type,
        '{}/{}_itot032_swadvec4m'.format(case_dir, experiment))
    data64_4m = Microhh(
        t,
        64,
        32,
        float_type,
        '{}/{}_itot064_swadvec4m'.format(case_dir, experiment))
    data128_4m = Microhh(
        t,
        128,
        64,
        float_type,
        '{}/{}_itot128_swadvec4m'.format(case_dir, experiment))
    data256_4m = Microhh(
        t,
        256,
        128,
        float_type,
        '{}/{}_itot256_swadvec4m'.format(case_dir, experiment))

    ref16_4m = Get_ref(data16_4m .x, data16_4m .xh,
                      data16_4m .z, data16_4m .zh, visc, time)
    ref32_4m = Get_ref(data32_4m .x, data32_4m .xh,
                      data32_4m .z, data32_4m .zh, visc, time)
    ref64_4m = Get_ref(data64_4m .x, data64_4m .xh,
                      data64_4m .z, data64_4m .zh, visc, time)
    ref128_4m = Get_ref(data128_4m.x, data128_4m.xh,
                       data128_4m.z, data128_4m.zh, visc, time)
    ref256_4m = Get_ref(data256_4m.x, data256_4m.xh,
                       data256_4m.z, data256_4m.zh, visc, time)

    err16_4m = Get_error(data16_4m, ref16_4m)
    err32_4m = Get_error(data32_4m, ref32_4m)
    err64_4m = Get_error(data64_4m, ref64_4m)
    err128_4m = Get_error(data128_4m, ref128_4m)
    err256_4m = Get_error(data256_4m, ref256_4m)

    errsu_4m = array(
        [err16_4m.u, err32_4m.u, err64_4m.u, err128_4m.u, err256_4m.u])
    errsw_4m = array(
        [err16_4m.w, err32_4m.w, err64_4m.w, err128_4m.w, err256_4m.w])
    errsp_4m = array(
        [err16_4m.p, err32_4m.p, err64_4m.p, err128_4m.p, err256_4m.p])

    print('errors p_4thm', errsp_4m)
    if(t > 0):
        print('convergence u_4thm',
              (log(errsu_4m[-1]) - log(errsu_4m[0])) / (log(dxs[-1]) - log(dxs[0])))
        print('convergence w_4thm',
              (log(errsw_4m[-1]) - log(errsw_4m[0])) / (log(dxs[-1]) - log(dxs[0])))
    print('convergence p_4thm',
          (log(errsp_4m[-1]) - log(errsp_4m[0])) / (log(dxs[-1]) - log(dxs[0])))

    # 4th order data
    data16_4th = Microhh(
        t,
        16,
        8,
        float_type,
        '{}/{}_itot016_swadvec4'.format(case_dir, experiment))
    data32_4th = Microhh(
        t,
        32,
        16,
        float_type,
        '{}/{}_itot032_swadvec4'.format(case_dir, experiment))
    data64_4th = Microhh(
        t,
        64,
        32,
        float_type,
        '{}/{}_itot064_swadvec4'.format(case_dir, experiment))
    data128_4th = Microhh(
        t,
        128,
        64,
        float_type,
        '{}/{}_itot128_swadvec4'.format(case_dir, experiment))
    data256_4th = Microhh(
        t,
        256,
        128,
        float_type,
        '{}/{}_itot256_swadvec4'.format(case_dir, experiment))

    ref16_4th = Get_ref(data16_4th .x, data16_4th .xh,
                       data16_4th .z, data16_4th .zh, visc, time)
    ref32_4th = Get_ref(data32_4th .x, data32_4th .xh,
                       data32_4th .z, data32_4th .zh, visc, time)
    ref64_4th = Get_ref(data64_4th .x, data64_4th .xh,
                       data64_4th .z, data64_4th .zh, visc, time)
    ref128_4th = Get_ref(data128_4th.x, data128_4th.xh,
                        data128_4th.z, data128_4th.zh, visc, time)
    ref256_4th = Get_ref(data256_4th.x, data256_4th.xh,
                        data256_4th.z, data256_4th.zh, visc, time)

    err16_4th = Get_error(data16_4th, ref16_4th)
    err32_4th = Get_error(data32_4th, ref32_4th)
    err64_4th = Get_error(data64_4th, ref64_4th)
    err128_4th = Get_error(data128_4th, ref128_4th)
    err256_4th = Get_error(data256_4th, ref256_4th)

    errsu_4th = array(
        [err16_4th.u, err32_4th.u, err64_4th.u, err128_4th.u, err256_4th.u])
    errsw_4th = array(
        [err16_4th.w, err32_4th.w, err64_4th.w, err128_4th.w, err256_4th.w])
    errsp_4th = array(
        [err16_4th.p, err32_4th.p, err64_4th.p, err128_4th.p, err256_4th.p])

    print('errors p_4th', errsp_4th)
    if(t > 0):
        print('convergence u_4th', (log(
            errsu_4th[-1]) - log(errsu_4th[0])) / (log(dxs[-1]) - log(dxs[0])))
        print('convergence w_4th', (log(
            errsw_4th[-1]) - log(errsw_4th[0])) / (log(dxs[-1]) - log(dxs[0])))
    print('convergence p_4th',
          (log(errsp_4th[-1]) - log(errsp_4th[0])) / (log(dxs[-1]) - log(dxs[0])))

    off2 = 0.01
    off4 = 0.002
    slope2 = off2 * (dxs[:] / dxs[0])**2.
    slope4 = off4 * (dxs[:] / dxs[0])**4.

    file_name = '{}/{}_{}.pdf'.format(case_dir, case_name, experiment)

    close('all')
    with PdfPages(file_name) as pdf:
        figure()
        if(t > 0):
            loglog(dxs, errsu_2nd, 'bo-', label="u_2nd")
            loglog(dxs, errsw_2nd, 'bv-', label="w_2nd")
            loglog(dxs, errsu_4m, 'go-', label="u_4thm")
            loglog(dxs, errsw_4m, 'gv-', label="w_4thm")
            loglog(dxs, errsu_4th, 'ro-', label="u_4th")
            loglog(dxs, errsw_4th, 'rv-', label="w_4th")
        loglog(dxs, errsp_2nd, 'b^-', label="p_2nd")
        loglog(dxs, errsp_4m, 'g^-', label="p_4thm")
        loglog(dxs, errsp_4th, 'r^-', label="p_4th")
        loglog(dxs, slope2, 'k--', label="2nd")
        loglog(dxs, slope4, 'k:', label="4th")
        legend(loc=0, frameon=False)
        xlabel('dx')
        ylabel('error')
        tight_layout()
        pdf.savefig()

        figure()
        subplot(121)
        pcolormesh(data256_2nd.x, data256_2nd.z,
                   data256_2nd.u[:, 0, :] - ref256_2nd.u[:, 0, :], rasterized=True)
        xlim(min(data256_2nd.xh), max(data256_2nd.xh))
        ylim(min(data256_2nd.z), max(data256_2nd.z))
        xlabel('x')
        ylabel('z')
        title('u err_2nd')
        colorbar()
        subplot(122)
        pcolormesh(data256_4th.x, data256_4th.z,
                   data256_4th.u[:, 0, :] - ref256_4th.u[:, 0, :], rasterized=True)
        xlim(min(data256_4th.xh), max(data256_4th.xh))
        ylim(min(data256_4th.z), max(data256_4th.z))
        xlabel('x')
        ylabel('z')
        title('u err_4th')
        colorbar()
        tight_layout()
        pdf.savefig()

        figure()
        subplot(121)
        pcolormesh(data256_2nd.x, data256_2nd.z,
                   data256_2nd.w[:, 0, :] - ref256_2nd.w[:, 0, :], rasterized=True)
        xlim(min(data256_2nd.xh), max(data256_2nd.xh))
        ylim(min(data256_2nd.z), max(data256_2nd.z))
        xlabel('x')
        ylabel('z')
        title('w err_2nd')
        colorbar()
        subplot(122)
        pcolormesh(data256_4th.x, data256_4th.z,
                   data256_4th.w[:, 0, :] - ref256_4th.w[:, 0, :], rasterized=True)
        xlim(min(data256_4th.x), max(data256_4th.x))
        ylim(min(data256_4th.zh), max(data256_4th.zh))
        xlabel('x')
        ylabel('z')
        title('w err_4th')
        colorbar()
        tight_layout()
        pdf.savefig()

        figure()
        subplot(121)
        pcolormesh(data256_2nd.x, data256_2nd.z,
                   data256_2nd.p[:, 0, :] - ref256_2nd.p[:, 0, :], rasterized=True)
        xlim(min(data256_2nd.x), max(data256_2nd.x))
        ylim(min(data256_2nd.z), max(data256_2nd.z))
        xlabel('x')
        ylabel('z')
        title('p err_2nd')
        colorbar()
        subplot(122)
        pcolormesh(data256_4th.x, data256_4th.z,
                   data256_4th.p[:, 0, :] - ref256_4th.p[:, 0, :], rasterized=True)
        xlim(min(data256_4th.x), max(data256_4th.x))
        ylim(min(data256_4th.z), max(data256_4th.z))
        xlabel('x')
        ylabel('z')
        title('p err_4th')
        colorbar()
        tight_layout()
        pdf.savefig()


def run_test(executable='microhh', prec='dp', mode='cpu', case_dir='.', experiment='local'):

    case_name = 'taylorgreen'

    mht.run_permutations(
            case_name, no_opts, opt_mpi, [dict_resolution, dict_order],
            executable=executable, mode=mode, case_dir=case_dir, experiment=experiment)

    plot(case_name, case_dir, experiment, prec)


if __name__ == '__main__':

    run_test()
