/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: cuda_intrinsics.h 1100 2026-10-10 16:13:45Z jasonp_sf $
--------------------------------------------------------------------*/

#if !defined(CUDA_INTRINSICS_H)
#define CUDA_INTRINSICS_H

typedef char int8;
typedef unsigned char uint8;
typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef long long int64;

/* 128-bit integers */

typedef struct {
	uint32 w[4];
} uint128;

/*------------------- Low-level functions ------------------------------*/

__device__ inline void
accum(uint32 &a0, uint32 &a1, uint32 &a2,
	uint32 b0, uint32 b1) {

	asm("sub.cc.u32 %0, %0, %3;   /* inline */   \n\t"
	    "subc.cc.u32 %1, %1, %4;   /* inline */   \n\t"
	    "subc.u32 %2, %2, %5;   /* inline */   \n\t"
		: "+r"(a0), "+r"(a1), "+r"(a2)
		: "r"(b0), "r"(b1), "r"(0) );
}

__device__ inline void
accum(uint64 &a0, uint64 &a1, uint64 &a2,
	uint64 b0, uint64 b1) {

	asm("sub.cc.u64 %0, %0, %3;   /* inline */   \n\t"
	    "subc.cc.u64 %1, %1, %4;   /* inline */   \n\t"
	    "subc.u64 %2, %2, %5;   /* inline */   \n\t"
		: "+l"(a0), "+l"(a1), "+l"(a2)
		: "l"(b0), "l"(b1), "l"((uint64)(0)) );
}

/*----------------- Squaring ----------------------------------------*/

__device__ inline uint64 
wide_sqr32(uint32 a)
{
	uint64 aa;

	asm("mul.wide.u32 %0, %1, %1; /* inline */ \n\t"
	    : "=l"(aa)
	    : "r"(a));

	return aa;
}

/* -------------------- Modular subtraction ------------------------*/

__device__ inline uint32 
modsub(uint32 a, uint32 b, uint32 p) 
{
	uint32 r;

	asm("{  \n\t"
	    ".reg .pred %pborrow;           \n\t"
	    ".reg .u32 %borrow;           \n\t"
	    "mov.b32 %borrow, 0;           \n\t"
	    "sub.cc.u32 %0, %1, %2;        \n\t"
	    "subc.u32 %borrow, %borrow, 0; \n\t"
	    "setp.ne.u32 %pborrow, %borrow, 0;  \n\t"
	    "@%pborrow add.u32 %0, %0, %3; \n\t"
	    "} \n\t"
	    : "=r"(r) : "r"(a), "r"(b), "r"(p) );

	return r;
}

__device__ inline uint64 
modsub(uint64 a, uint64 b, uint64 p) 
{
	uint64 r;

	asm("{  \n\t"
	    ".reg .pred %pborrow;           \n\t"
	    ".reg .u64 %borrow;           \n\t"
	    "mov.b64 %borrow, 0;           \n\t"
	    "sub.cc.u64 %0, %1, %2;        \n\t"
	    "subc.u64 %borrow, %borrow, 0; \n\t"
	    "setp.ne.u64 %pborrow, %borrow, 0;  \n\t"
	    "@%pborrow add.u64 %0, %0, %3; \n\t"
	    "} \n\t"
	    : "=l"(r) : "l"(a), "l"(b), "l"(p) );

	return r;
}

/*------------------------------- GCD --------------------------------*/
__device__ inline uint32
gcd(uint32 x, uint32 y) {

	/* assumes x and y are odd and nonzero */

	uint32 u = x;
	uint32 v = y;

	do {
		v >>= __ffs(v) - 1;

		x = umin(u, v);
		y = umax(u, v);
		u = x;
		v = y - x;
	} while (v != 0);

	return u;
}

__device__ inline uint64
gcd(uint64 x, uint64 y) {

	/* assumes x and y are odd and nonzero */

	uint64 u = x;
	uint64 v = y;

	do {
		v >>= __ffsll(v) - 1;

		x = ullmin(u, v);
		y = ullmax(u, v);
		u = x;
		v = y - x;
	} while (v != 0);

	return u;
}

/*-------------------------- Modular inverse -------------------------*/

__device__ inline uint32 
modinv(uint32 a, uint32 p) {

	uint32 ps1, ps2, dividend, divisor, rem, q, t;
	uint32 parity;

	q = 1; rem = a; dividend = p; divisor = a;
	ps1 = 1; ps2 = 0; parity = 0;

	while (divisor > 1) {
		rem = dividend - divisor;
		t = rem - divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t;
		if (rem >= divisor) {
			q = dividend / divisor;
			rem = dividend - q * divisor;
			q *= ps1;
		} } } } } } } } }

		q += ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
	}
	
	if (parity == 0)
		return ps1;
	else
		return p - ps1;
}

__device__ inline uint64 
modinv(uint64 a, uint64 p) {

	uint64 ps1, ps2, dividend, divisor, rem, q, t;
	uint32 parity;

	q = 1; rem = a; dividend = p; divisor = a;
	ps1 = 1; ps2 = 0; parity = 0;

	while (divisor > 1) {
		rem = dividend - divisor;
		t = rem - divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t;
		if (rem >= divisor) {
			q = dividend / divisor;
			rem = dividend - q * divisor;
			q *= ps1;
		} } } } } } } } }

		q += ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
	}
	
	if (parity == 0)
		return ps1;
	else
		return p - ps1;
}

/*------------------- Montgomery arithmetic --------------------------*/
__device__ inline uint32 
montmul(uint32 a, uint32 b, uint32 n, uint32 w) {

	uint32 acc0, acc1, acc2 = 0;
	uint32 q;
	uint32 prod_lo, prod_hi;

	acc0 = a * b;
	acc1 = __umulhi(a, b);

	q = acc0 * w;

	prod_lo = q * n;
	prod_hi = __umulhi(q, n);

	accum(acc0, acc1, acc2, prod_lo, prod_hi);

	if (acc2)
		return acc1 + n;
	else
		return acc1;
}

__device__ inline uint64 
montmul(uint64 a, uint64 b, uint64 n, uint64 w) {

	uint64 acc0, acc1, acc2 = 0;
	uint64 q;
	uint64 prod_lo, prod_hi;

	acc0 = a * b;
	acc1 = __umul64hi(a, b);

	q = acc0 * w;

	prod_lo = q * n;
	prod_hi = __umul64hi(q, n);

	accum(acc0, acc1, acc2, prod_lo, prod_hi);

	if (acc2)
		return acc1 + n;
	else
		return acc1;
}

/*-------------------------- Mod -------------------------------------*/
/* The algorithm is from Ernst Mayer, "Efficient Long Division via
   Montgomery Multiply. The quantity t below is 
   2^(bits_in_numerator+bits_in_word) */

__device__ inline uint32
mod(uint64 num, uint32 t, uint32 n, uint32 w) {

	uint32 x, rem, tmp;

	x = (uint32)num;
	tmp = x * w;
	rem = __umulhi(tmp, n);
	x = (uint32)(num >> 32);
	tmp = x - rem;
	tmp = tmp * w;
	if (rem > x)
		tmp++;
	rem = __umulhi(tmp, n);
	rem = montmul(rem, t, n, w);

	if (rem == 0)
		return 0;
	else
		return n - rem;
}

__device__ inline uint32
mod(uint128 num, uint32 t, uint32 n, uint32 w) {

	uint32 x, rem, tmp;

	x = num.w[0];
	tmp = x * w;
	rem = __umulhi(tmp, n);
	x = num.w[1];
	tmp = x - rem;
	tmp = tmp * w;
	if (rem > x)
		tmp++;
	rem = __umulhi(tmp, n);
	x = num.w[2];
	tmp = x - rem;
	tmp = tmp * w;
	if (rem > x)
		tmp++;
	rem = __umulhi(tmp, n);
	x = num.w[3];
	tmp = x - rem;
	tmp = tmp * w;
	if (rem > x)
		tmp++;
	rem = __umulhi(tmp, n);
	rem = montmul(rem, t, n, w);

	if (rem == 0)
		return 0;
	else
		return n - rem;
}

__device__ inline uint64
mod(uint128 num, uint64 t, uint64 n, uint64 w) {

	uint64 x, rem, hi, tmp;

	x = (uint64)num.w[1] << 32 | num.w[0];
	tmp = x * w;
	hi = __umul64hi(tmp, n);
	x = (uint64)num.w[3] << 32 | num.w[2];
	tmp = x - hi;
	tmp = tmp * w;
	if (hi > x)
		tmp++;
	hi = __umul64hi(tmp, n);
	rem = montmul(hi, t, n, w);

	if (rem == 0)
		return 0;
	else
		return n - rem;
}

#endif /* !defined(CUDA_INTRINSICS_H) */

