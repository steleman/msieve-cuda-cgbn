/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id: poly_skew.c 1102 2026-10-30 14:36:01Z jasonp_sf $
--------------------------------------------------------------------*/

#include "poly_skew.h"
#include <thread.h>
#include <sys/time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef HAVE_CUDA
#include <stage2/root_sieve_gpu.h>
#endif

/* root optimization: in GPU builds, by the engine of root_sieve_gpu.c
   (the batched root sieve, with its optimize_basic() calls and line
   sieves on the GPU); with MSIEVE_GPU_ROOTOPT=0, or without a usable
   GPU, on the CPU (poly_rootopt_run, msieve's own code) */
#ifdef HAVE_CUDA
static ro_gpu_t *ro_engine;
#endif

/* MPI (polynomial selection on several processes, possibly nodes):
   each process searches its share and writes its own output files,
   rank 0 under the usual names, rank r > 0 as <name>.<ext>.mpiNN (as
   the logs). At the end rank 0 collects the other processes' files
   and best polynomials over MPI (no shared file system needed) */
static uint32_t poly_rank, poly_nranks = 1;

static void poly_out_name(char *buf, msieve_obj *obj, const char *ext) {
	if (poly_rank == 0)
		sprintf(buf, "%s.%s", obj->savefile.name, ext);
	else
		sprintf(buf, "%s.%s.mpi%02u", obj->savefile.name, ext,
				poly_rank);
}

static void rootopt_run(poly_rootopt_t *data, mpz_t *alg_coeffs,
			mpz_t *rat_coeffs, double sizeopt_norm,
			double projective_alpha) {
#ifdef HAVE_CUDA
	if (ro_engine != NULL) {
		ro_gpu_submit(ro_engine, alg_coeffs, rat_coeffs,
				sizeopt_norm, projective_alpha);
		return;
	}
#endif
	poly_rootopt_run(data, alg_coeffs, rat_coeffs, sizeopt_norm,
			projective_alpha);
}

static void rootopt_start(msieve_obj *obj, poly_rootopt_t *proto) {
#ifdef HAVE_CUDA
	const char *e = getenv("MSIEVE_GPU_ROOTOPT");

	if (e == NULL || strcmp(e, "0") != 0)
		ro_engine = ro_gpu_init(obj, proto);
#else
	(void)obj;
	(void)proto;
#endif
}

/* every root optimization submitted so far is done */
static void rootopt_finish(void) {
#ifdef HAVE_CUDA
	ro_gpu_finish(ro_engine);
	ro_engine = NULL;
#endif
}

/* stage 2 (size and root optimization) can run on several threads
   (MSIEVE_STAGE2_THREADS, default: online CPUs). Each thread has its own
   poly_sizeopt_t / poly_rootopt_t; the outputs they share (the .ms
   and .p files, the best-poly list, the measurement log) are written
   under this lock */
static mutex_t stage2_out_lock;

static FILE *sizeopt_log_file(void);

/*------------------------------------------------------------------*/
/* callback for root optimization */

typedef struct {
	FILE *all_poly_file;
	poly_config_t *config;
	double report_min_e;	/* printed and written from this E (msieve's
				   minimum), or when a new best */
} rootopt_callback_data_t;

static void
rootopt_callback(void *extra, uint32 degree, 
		mpz_t * coeff1, mpz_t * coeff2,
		double skewness, double size_score,
		double root_score, double combined_score,
		uint32 num_real_roots)
{
	rootopt_callback_data_t *data = (rootopt_callback_data_t *)extra;
	poly_config_t *config = data->config;
	FILE *lf = sizeopt_log_file();

	mutex_lock(&stage2_out_lock);

	/* measurement (MSIEVE_SIZEOPT_LOG): every root optimization result,
	   identified by its leading and rational coefficients, which root
	   optimization does not change, so that it can be matched with
	   its size-optimized survivor ("P" line) */
	if (lf != NULL) {
		gmp_fprintf(lf, "R %Zd %Zd %le %le %lf\n", coeff1[degree],
				coeff2[1], combined_score, size_score,
				root_score);
		fflush(lf);
	}

	/* polynomials below msieve's minimum E are only kept if they are the
	   best so far (MSIEVE_POLY_KEEP_E): a c110 kept 485,371 of them, and
	   printed and wrote every one */
	if (combined_score < data->report_min_e &&
	    combined_score <= config->heap[0]->combined_score) {
		mutex_unlock(&stage2_out_lock);
		return;
	}

	poly_select_t poly;
	memset(&poly, 0, sizeof(poly_select_t));
	mpz_poly_t *rpoly = &poly.rpoly;
	mpz_poly_t *apoly = &poly.apoly;
	mpz_poly_init(rpoly);
	mpz_poly_init(apoly);

	rpoly->degree = 1;
	for (uint32_t i = 0; i <= 1; i++)
		mpz_set(rpoly->coeff[i], coeff2[i]);

	apoly->degree = degree;
	for (uint32_t i = 0; i <= degree; i++)
		mpz_set(apoly->coeff[i], coeff1[i]);

	poly.root_score = root_score;
	poly.size_score = size_score;
	poly.combined_score = combined_score;
	poly.skewness = skewness;

	printf("save %le %.4lf %.2lf %le rroots %u\n", size_score,
			root_score, skewness, combined_score,
			num_real_roots);

	fprintf(data->all_poly_file, 
		"# norm %le alpha %lf e %.3le rroots %u\nskew: %.2lf\n", 
		size_score, root_score, combined_score, 
		num_real_roots, skewness);
	for (uint32_t i = 0; i <= degree; i++)
		gmp_fprintf(data->all_poly_file, "c%u: %Zd\n", i, coeff1[i]);
	for (uint32_t i = 0; i <= 1; i++)
		gmp_fprintf(data->all_poly_file, "Y%u: %Zd\n", i, coeff2[i]);
	fflush(data->all_poly_file);

	save_poly(config, &poly);
	mpz_poly_free(rpoly);
	mpz_poly_free(apoly);
	mutex_unlock(&stage2_out_lock);
}

/*------------------------------------------------------------------*/
/* callbacks for size optimization */

typedef struct {
	poly_rootopt_t *rootopt;
	rootopt_callback_data_t *rootopt_callback;
} sizeopt_callback_data_t;

/* measurement only: MSIEVE_SIZEOPT_LOG=<file> logs every size
   optimization survivor (time, stage 1 hits so far, norm, alpha) and
   the stage 1 hit count every 100000 hits */
static FILE *sizeopt_log;
static uint64_t sizeopt_log_hits;

static FILE *sizeopt_log_file(void) {
	static bool init;
	if (!init) {
		const char *e = getenv("MSIEVE_SIZEOPT_LOG");
		init = true;
		if (e != NULL && poly_rank == 0) {
			sizeopt_log = fopen(e, "a");
		}
		else if (e != NULL) {
			char name[1024];

			snprintf(name, sizeof(name), "%s.mpi%02u", e,
					poly_rank);
			sizeopt_log = fopen(name, "a");
		}
	}
	return sizeopt_log;
}

static double sizeopt_log_now(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* the wall-clock end of polynomial selection (0: none): past it,
   size-optimized polynomials are not root-optimized any more, so the
   selection ends with the root optimizations already under way. After
   stage 1 stopped, the polynomials still on their way through size
   optimization kept root optimization busy for minutes */
double poly_report_norm;
static double polysel_wall_end;
double rootopt_wall_end;
uint64_t rootopt_dropped;
static uint64_t polysel_skipped;

/* MSIEVE_ROOTOPT_TOPK=<K>: root-optimize a size-optimized polynomial
   only if its size score (norm * exp(projective alpha), the stage 2
   bound's) is among the K best seen so far, as CADO-NFS root-optimizes
   only its nrkeep best. Root optimization is most of the GPU's work in
   polynomial selection; fewer of them leave it to stage 1, and the
   selection ends at its time limit instead of minutes past it. The
   size score hardly predicts the final E among the survivors of the
   stage 2 bound (c120: Spearman -0.09), so this is a random subset in
   effect: it does not find better polynomials, it bounds the work.
   0: all; default 48 in GPU builds (where the short time limits are),
   0 otherwise. */
static uint32_t topk_k;
static double *topk_best;	/* the K best scores so far, ascending */
static uint32_t topk_n;
static uint64_t topk_skipped;
static mutex_t topk_lock;

static void topk_init(void) {
	const char *e = getenv("MSIEVE_ROOTOPT_TOPK");

#ifdef HAVE_CUDA
	topk_k = (e != NULL) ? (uint32_t)strtoul(e, NULL, 10) : 48;
#else
	topk_k = (e != NULL) ? (uint32_t)strtoul(e, NULL, 10) : 0;
#endif
	topk_n = 0;
	topk_skipped = 0;
	free(topk_best);
	topk_best = NULL;
	if (topk_k > 0) {
		topk_best = (double *)xmalloc(topk_k * sizeof(double));
		mutex_init(&topk_lock);
	}
}

/* true if the polynomial is to be root-optimized */
static bool topk_admit(double score) {
	bool admit;

	if (topk_k == 0)
		return true;
	mutex_lock(&topk_lock);
	admit = (topk_n < topk_k || score < topk_best[topk_k - 1]);
	if (admit) {
		uint32_t i = (topk_n < topk_k) ? topk_n++ : topk_k - 1;

		while (i > 0 && topk_best[i - 1] > score) {
			topk_best[i] = topk_best[i - 1];
			i--;
		}
		topk_best[i] = score;
	}
	else {
		topk_skipped++;
	}
	mutex_unlock(&topk_lock);
	return admit;
}

static void sizeopt_callback(uint32 deg, mpz_t *alg_coeffs, mpz_t *rat_coeffs, 
				double sizeopt_norm, double projective_alpha, 
				void *extra)
{
	sizeopt_callback_data_t *callback = (sizeopt_callback_data_t *)extra;
	FILE *lf;

	if (polysel_wall_end != 0 && sizeopt_log_now() > polysel_wall_end) {
		__atomic_fetch_add(&polysel_skipped, 1, __ATOMIC_RELAXED);
		return;
	}
	lf = sizeopt_log_file();

	if (lf != NULL) {
		mutex_lock(&stage2_out_lock);
		gmp_fprintf(lf, "S %.3f %" PRIu64 " %Zd %le %.3lf\n",
				sizeopt_log_now(), sizeopt_log_hits,
				alg_coeffs[deg], sizeopt_norm,
				projective_alpha);
		/* the survivor itself, as a .ms line after "P " */
		{
			fprintf(lf, "P ");
			for (uint32_t i = deg; (int32_t)i >= 0; i--)
				gmp_fprintf(lf, "%Zd ", alg_coeffs[i]);
			gmp_fprintf(lf, "%Zd %Zd %.2lf %le\n", rat_coeffs[1],
					rat_coeffs[0], projective_alpha,
					exp(projective_alpha) * sizeopt_norm);
		}
		fflush(lf);
		mutex_unlock(&stage2_out_lock);
	}

	if (!topk_admit(sizeopt_norm * exp(projective_alpha)))
		return;
	rootopt_run(callback->rootopt, alg_coeffs, 
			rat_coeffs, sizeopt_norm, projective_alpha);
}

static void sizeopt_callback_log(uint32 deg, mpz_t *alg_coeffs, mpz_t *rat_coeffs, 
				double sizeopt_norm, double projective_alpha, 
				void *extra)
{
	FILE *mfile = (FILE *)extra;

	mutex_lock(&stage2_out_lock);
	for (uint32_t i = deg; (int32_t)i >= 0; i--)
		gmp_fprintf(mfile, "%Zd ", alg_coeffs[i]);

	gmp_fprintf(mfile, "%Zd %Zd %.2lf %le\n", rat_coeffs[1], 
				rat_coeffs[0], projective_alpha, 
				exp(projective_alpha) * sizeopt_norm);
	fflush(mfile);
	mutex_unlock(&stage2_out_lock);
}

/*------------------------------------------------------------------*/
/* a polynomial size-optimized on the GPU: the same bound test as
   poly_sizeopt_run, then on to whatever follows size optimization */

static void stage1_sizeopt_callback(void *extra, uint32 deg,
				mpz_t *alg_coeffs, mpz_t *rat_coeffs,
				double sizeopt_norm, double projective_alpha)
{
	poly_sizeopt_t *data = (poly_sizeopt_t *)extra;

	if (sizeopt_norm * exp(projective_alpha) <= data->max_sizeopt_norm)
		data->callback(deg, alg_coeffs, rat_coeffs, sizeopt_norm,
				projective_alpha, data->callback_data);
}

/* MSIEVE_GPU_SIZEOPT=0 keeps size optimization on the CPU */
static bool gpu_sizeopt_wanted(uint32_t degree) {
	const char *e = getenv("MSIEVE_GPU_SIZEOPT");

	if (degree != 4 && degree != 5)
		return false;
	return (e == NULL || strcmp(e, "0") != 0);
}

/*------------------------------------------------------------------*/
/* callbacks for stage 1 */

static void stage1_callback(mpz_t ad, mpz_t p, mpz_t m, void *extra) {
	
	poly_sizeopt_t *data = (poly_sizeopt_t *)extra;
	FILE *lf = sizeopt_log_file();

	if (lf != NULL) {
		mutex_lock(&stage2_out_lock);
		if (++sizeopt_log_hits % 100000 == 0) {
			fprintf(lf, "H %.3f %" PRIu64 "\n",
				sizeopt_log_now(), sizeopt_log_hits);
			fflush(lf);
		}
		mutex_unlock(&stage2_out_lock);
	}
	poly_sizeopt_run(data, ad, p, m);
}

typedef struct {
	FILE *mfile;
	bool verbose;           /* -v: echo each hit to stdout too */
} stage1_log_t;

static void stage1_callback_log(mpz_t ad, mpz_t p, mpz_t m, void *extra) {
	
	const stage1_log_t *log = (const stage1_log_t *)extra;
	if (log->verbose)
		gmp_printf("%Zd %Zd %Zd\n", ad, p, m);
	gmp_fprintf(log->mfile, "%Zd %Zd %Zd\n", ad, p, m);
	fflush(log->mfile);
}

/*------------------------------------------------------------------*/
/* stage 2 workers: worker 0 is find_poly_core's own sizeopt_data /
   rootopt_data; workers 1.. are copies with the same parameters */

typedef struct {
	poly_sizeopt_t sizeopt;
	poly_rootopt_t rootopt;
	sizeopt_callback_data_t sizeopt_cb;
} stage2_worker_t;

typedef struct {
	uint32_t num;
	stage2_worker_t *w;		/* num - 1 extra workers */
	poly_sizeopt_t **sizeopt;	/* num entries, [0] = the original */
	poly_rootopt_t **rootopt;	/* num entries, or NULL */
} stage2_workers_t;

/* default: one stage 2 thread per online CPU (shared by the node's MPI
   processes) */
static uint32_t stage2_thread_count(uint32_t local_size) {
	const char *e = getenv("MSIEVE_STAGE2_THREADS");
	long k;

	if (e != NULL) {
		k = strtol(e, NULL, 10);
	}
	else {
#if defined(WIN32) || defined(_WIN64)
		SYSTEM_INFO si;
		GetSystemInfo(&si);
		k = (long)si.dwNumberOfProcessors;
#else
		k = sysconf(_SC_NPROCESSORS_ONLN);
#endif
		k /= MAX(1, (long)local_size);
	}
	return (uint32_t)MAX(1, MIN(k, 256));
}

static void stage2_workers_init(stage2_workers_t *s, uint32_t num,
			msieve_obj *obj, mpz_t n,
			poly_sizeopt_t *sizeopt0,	/* NULL: no size opt */
			poly_rootopt_t *rootopt0,	/* NULL: no root opt */
			sizeopt_callback_t sizeopt_cb,
			void *sizeopt_cb_data,	/* when !rootopt0 */
			rootopt_callback_data_t *rootopt_cb_data) {
	s->num = num;
	s->w = (stage2_worker_t *)xcalloc(MAX(1, num - 1),
					sizeof(stage2_worker_t));
	s->sizeopt = (poly_sizeopt_t **)xcalloc(num, sizeof(void *));
	s->rootopt = (poly_rootopt_t **)xcalloc(num, sizeof(void *));
	s->sizeopt[0] = sizeopt0;
	s->rootopt[0] = rootopt0;

	for (uint32_t i = 1; i < num; i++) {
		stage2_worker_t *w = s->w + i - 1;

		if (rootopt0 != NULL) {
			poly_rootopt_t *r = &w->rootopt;

			poly_rootopt_init(r, obj, rootopt_callback,
					rootopt_cb_data);
			mpz_set(r->gmp_N, n);
			r->degree = rootopt0->degree;
			r->murphy_p_bound = rootopt0->murphy_p_bound;
			r->max_sizeopt_norm = rootopt0->max_sizeopt_norm;
			r->min_e = rootopt0->min_e;
			r->min_e_bernstein = rootopt0->min_e_bernstein;
			s->rootopt[i] = r;
		}
		if (sizeopt0 != NULL) {
			poly_sizeopt_t *z = &w->sizeopt;

			if (rootopt0 != NULL) {
				w->sizeopt_cb.rootopt = &w->rootopt;
				w->sizeopt_cb.rootopt_callback =
							rootopt_cb_data;
				poly_sizeopt_init(z, sizeopt_cb,
						&w->sizeopt_cb);
			}
			else {
				poly_sizeopt_init(z, sizeopt_cb,
						sizeopt_cb_data);
			}
			mpz_set(z->gmp_N, n);
			z->degree = sizeopt0->degree;
			z->max_stage1_norm = sizeopt0->max_stage1_norm;
			z->max_sizeopt_norm = sizeopt0->max_sizeopt_norm;
			s->sizeopt[i] = z;
		}
	}
}

static void stage2_workers_free(stage2_workers_t *s) {
	for (uint32_t i = 1; i < s->num; i++) {
		if (s->sizeopt[i] != NULL)
			poly_sizeopt_free(s->sizeopt[i]);
		if (s->rootopt[i] != NULL)
			poly_rootopt_free(s->rootopt[i]);
	}
	free(s->w);
	free(s->sizeopt);
	free(s->rootopt);
}

/* one line of a .m (size opt) or .ms (root opt) file, run on
   whichever stage 2 thread takes it */

typedef struct {
	stage2_workers_t *s;
	bool root;			/* 0: size opt, 1: root opt */
	mpz_t c[MAX_POLY_DEGREE + 1];	/* ad p m, or alg coeffs */
	mpz_t r[2];			/* rat coeffs (root opt) */
} stage2_line_t;

static void stage2_line_run(void *data, int threadid) {
	stage2_line_t *l = (stage2_line_t *)data;

	if (l->root)
		rootopt_run(l->s->rootopt[threadid], l->c, l->r, 0, 0);
	else
		poly_sizeopt_run(l->s->sizeopt[threadid],
				l->c[0], l->c[1], l->c[2]);
}

static void stage2_line_free(void *data, int threadid) {
	stage2_line_t *l = (stage2_line_t *)data;

	(void)threadid;
	for (uint32_t i = 0; i <= MAX_POLY_DEGREE; i++)
		mpz_clear(l->c[i]);
	mpz_clear(l->r[0]);
	mpz_clear(l->r[1]);
	free(l);
}

static stage2_line_t *stage2_line_new(stage2_workers_t *s, bool root) {
	stage2_line_t *l = (stage2_line_t *)xmalloc(sizeof(stage2_line_t));

	l->s = s;
	l->root = root;
	for (uint32_t i = 0; i <= MAX_POLY_DEGREE; i++)
		mpz_init(l->c[i]);
	mpz_init(l->r[0]);
	mpz_init(l->r[1]);
	return l;
}

static void stage2_line_submit(struct threadpool *pool, stage2_line_t *l) {
	task_control_t t = {
		.init = NULL,
		.run = stage2_line_run,
		.shutdown = stage2_line_free,
		.data = l,
	};
	threadpool_add_task(pool, &t, 1);
}

/*------------------------------------------------------------------*/
#ifdef HAVE_MPI
/*------------------------------------------------------------------*/
/* rank r > 0 sends its <name>.<ext>.mpiNN to rank 0, which appends it to
   <name>.<ext>, in rank order; the sent file is removed */

#define MERGE_CHUNK (1 << 22)
#define TAG_FILE 7101
#define TAG_BEST 7102

static void
poly_mpi_merge_file(msieve_obj *obj, const char *ext)
{
	char name[1024];
	char *chunk = (char *)xmalloc(MERGE_CHUNK);
	uint64_t total = 0;

	poly_out_name(name, obj, ext);
	if (poly_rank == 0) {
		FILE *f = fopen(name, "a");

		if (f == NULL) {
			printf("error: cannot open %s\n", name);
			exit(-1);
		}
		for (uint32_t r = 1; r < poly_nranks; r++) {
			for (;;) {
				int len = 0;

				MPI_TRY(MPI_Recv(&len, 1, MPI_INT, (int)r,
					TAG_FILE, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE))
				if (len == 0)
					break;
				MPI_TRY(MPI_Recv(chunk, len, MPI_CHAR, (int)r,
					TAG_FILE, MPI_COMM_WORLD,
					MPI_STATUS_IGNORE))
				fwrite(chunk, 1, (size_t)len, f);
				total += (uint64_t)len;
			}
		}
		fclose(f);
		logprintf(obj, "polynomial selection: %.1f kB of %s from "
				"%u other processes\n", total / 1024.0, ext,
				poly_nranks - 1);
	}
	else {
		FILE *f = fopen(name, "r");
		int len = 0;

		while (f != NULL &&
		       (len = (int)fread(chunk, 1, MERGE_CHUNK, f)) > 0) {
			MPI_TRY(MPI_Send(&len, 1, MPI_INT, 0, TAG_FILE,
					MPI_COMM_WORLD))
			MPI_TRY(MPI_Send(chunk, len, MPI_CHAR, 0, TAG_FILE,
					MPI_COMM_WORLD))
		}
		len = 0;
		MPI_TRY(MPI_Send(&len, 1, MPI_INT, 0, TAG_FILE,
				MPI_COMM_WORLD))
		if (f != NULL) {
			fclose(f);
			remove(name);
		}
	}
	free(chunk);
}

/* every process's best polynomial to rank 0, exactly (hex), where
   save_poly() keeps the best of them for the .fb file */
static void
poly_mpi_merge_best(msieve_obj *obj, poly_config_t *config, uint32 degree)
{
	(void)obj;
	if (poly_rank != 0) {
		poly_select_t *b = config->heap[0];
		size_t cap = 65536, len = 0;
		char *s = (char *)xmalloc(cap);
		int n;

		if (b->rpoly.degree == 0) {
			len = (size_t)sprintf(s, "0\n");
		}
		else {
			len = (size_t)sprintf(s, "%u %a %a %a %a %u\n",
				b->apoly.degree, b->skewness, b->size_score,
				b->root_score, b->combined_score,
				b->num_real_roots);
			for (uint32_t i = 0; i <= b->apoly.degree; i++)
				len += (size_t)gmp_snprintf(s + len, cap - len,
						"%Zx\n", b->apoly.coeff[i]);
			for (uint32_t i = 0; i <= 1; i++)
				len += (size_t)gmp_snprintf(s + len, cap - len,
						"%Zx\n", b->rpoly.coeff[i]);
		}
		n = (int)len + 1;
		MPI_TRY(MPI_Send(&n, 1, MPI_INT, 0, TAG_BEST, MPI_COMM_WORLD))
		MPI_TRY(MPI_Send(s, n, MPI_CHAR, 0, TAG_BEST, MPI_COMM_WORLD))
		free(s);
		return;
	}

	for (uint32_t r = 1; r < poly_nranks; r++) {
		int n = 0;
		char *s, *t;
		uint32_t deg, nrr;
		poly_select_t p;

		MPI_TRY(MPI_Recv(&n, 1, MPI_INT, (int)r, TAG_BEST,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE))
		s = (char *)xmalloc((size_t)n);
		MPI_TRY(MPI_Recv(s, n, MPI_CHAR, (int)r, TAG_BEST,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE))
		memset(&p, 0, sizeof(p));
		if (sscanf(s, "%u %la %la %la %la %u", &deg, &p.skewness,
				&p.size_score, &p.root_score,
				&p.combined_score, &nrr) == 6 &&
		    deg == degree) {
			mpz_poly_init(&p.rpoly);
			mpz_poly_init(&p.apoly);
			p.num_real_roots = nrr;
			p.apoly.degree = deg;
			p.rpoly.degree = 1;
			t = strchr(s, '\n') + 1;
			for (uint32_t i = 0; i <= deg; i++) {
				gmp_sscanf(t, "%Zx", p.apoly.coeff[i]);
				t = strchr(t, '\n') + 1;
			}
			for (uint32_t i = 0; i <= 1; i++) {
				gmp_sscanf(t, "%Zx", p.rpoly.coeff[i]);
				t = strchr(t, '\n') + 1;
			}
			save_poly(config, &p);
			mpz_poly_free(&p.rpoly);
			mpz_poly_free(&p.apoly);
		}
		free(s);
	}
}
#endif

void find_poly_core(msieve_obj *obj, mpz_t n,
			poly_param_t *params,
			poly_config_t *config,
			uint32 degree) {

	poly_stage1_t stage1_data;
	poly_sizeopt_t sizeopt_data;
	poly_rootopt_t rootopt_data;
	sizeopt_callback_data_t sizeopt_callback_data;
	rootopt_callback_data_t rootopt_callback_data;
	char buf[2048];
	stage1_log_t stage1_log;
	FILE *stage1_outfile = NULL;
	FILE *sizeopt_outfile = NULL;
	const char *lower_limit = NULL;
	const char *upper_limit = NULL;
	stage2_workers_t workers;
	uint32_t num_stage2;
	struct threadpool *stage2_pool = NULL;
	uint32_t local_size = 1;
	uint64_t line_index = 0;

#ifdef HAVE_MPI
	poly_rank = obj->mpi_rank;
	poly_nranks = MAX(1, obj->mpi_size);
	local_size = obj->mpi_local_size;
	if (poly_nranks > 1)
		logprintf(obj, "polynomial selection: MPI process %u of %u "
				"(%u on this node)\n", poly_rank, poly_nranks,
				local_size);
#endif
	num_stage2 = stage2_thread_count(local_size);
	mutex_init(&stage2_out_lock);
	sizeopt_log_file();	/* open the measurement log, if any */
	memset(&workers, 0, sizeof(workers));

	/* make sure the configured stages have the bounds that
	   they need. We only have to check maximum bounds, since
	   if not provided then no polynomials will be found */

	if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) &&
	    params->stage1_norm == 0) {
		printf("error: stage 1 bound not provided\n");
		exit(-1);
	}
	if (((obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) ||
	     (obj->flags & MSIEVE_FLAG_NFS_POLYROOT)) &&
	    params->stage2_norm == 0) {
		printf("error: stage 2 size bound not provided\n");
		exit(-1);
	}
	if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) &&
	    (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) &&
	    !(obj->flags & MSIEVE_FLAG_NFS_POLYSIZE)) {
		printf("error: middle polyselect stage missing\n");
		exit(-1);
	}

	/* parse arguments */

	if (obj->nfs_args != NULL) {
		const char *tmp = strstr(obj->nfs_args, "min_coeff=");
		if (tmp != NULL)
			lower_limit = tmp + 10;

		tmp = strstr(obj->nfs_args, "max_coeff=");
		if (tmp != NULL)
			upper_limit = tmp + 10;

		/* old-style 'X,Y' format */
		tmp = strchr(obj->nfs_args, ',');
		if (tmp != NULL) {
			lower_limit = tmp - 1;
			while (lower_limit > obj->nfs_args &&
				isdigit(lower_limit[-1])) {
				lower_limit--;
			}
			upper_limit = tmp + 1;
		}

	}

	/* set up stage 1 */

	if (obj->flags & MSIEVE_FLAG_NFS_POLY1) {

		double coeff_scale = 3.0;

		/* if we're doing both stage 1 and size optimization, 
		   every hit in stage 1 is immediately forwarded. For 
		   stage 1 alone, all the stage 1 hits are buffered
		   to file first */

		if (obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) {
			poly_stage1_init(&stage1_data, 
					stage1_callback, &sizeopt_data);
		}
		else {
			poly_out_name(buf, obj, "m");
			stage1_outfile = fopen(buf, "a");
			if (stage1_outfile == NULL) {
				printf("error: cannot open poly1 file\n");
				exit(-1);
			}
			stage1_log.mfile = stage1_outfile;
			stage1_log.verbose = (obj->flags &
					MSIEVE_FLAG_LOG_TO_STDOUT) ? true : false;
			poly_stage1_init(&stage1_data, stage1_callback_log, 
					&stage1_log);
		}

		/* fill stage 1 data */

		mpz_set(stage1_data.gmp_N, n);
		stage1_data.degree = degree;
		stage1_data.norm_max = params->stage1_norm;
		stage1_data.deadline = params->deadline;
		polysel_wall_end = params->deadline ?
			sizeopt_log_now() + params->deadline : 0;
		rootopt_wall_end = polysel_wall_end;
		rootopt_dropped = 0;
		polysel_skipped = 0;
		topk_init();

		if (lower_limit != NULL)
			gmp_sscanf(lower_limit, "%Zd",
					stage1_data.gmp_high_coeff_begin);
		else
			mpz_set_ui(stage1_data.gmp_high_coeff_begin, 
					(unsigned long)1);

		if (upper_limit != NULL)
			gmp_sscanf(upper_limit, "%Zd",
					stage1_data.gmp_high_coeff_end);
		else
			mpz_set_d(stage1_data.gmp_high_coeff_end,
				pow(mpz_get_d(stage1_data.gmp_N), 1.0 / 
					(double)(degree * (degree - 1))) / 
				coeff_scale );

		if (stage1_data.deadline != 0)
			logprintf(obj, "time limit set to %.2f CPU-hours\n",
				stage1_data.deadline / 3600.0);

		logprintf(obj, "searching leading coefficients from "
				"%.0lf to %.0lf\n",
				mpz_get_d(stage1_data.gmp_high_coeff_begin),
				mpz_get_d(stage1_data.gmp_high_coeff_end));
	}

	/* set up size optimization */

	if (obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) {

		/* if we're doing both size and root optimization, 
		   every size opt hit in is immediately forwarded. For 
		   size optimization alone, all the hits are buffered
		   to file first */

		if (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) {
			sizeopt_callback_data.rootopt = &rootopt_data;
			sizeopt_callback_data.rootopt_callback = &rootopt_callback_data;

			poly_sizeopt_init(&sizeopt_data, 
					sizeopt_callback, &sizeopt_callback_data);
		}
		else {
			poly_out_name(buf, obj, "ms");
			sizeopt_outfile = fopen(buf, "a");
			if (sizeopt_outfile == NULL) {
				printf("error: cannot open sizeopt file\n");
				exit(-1);
			}
			poly_sizeopt_init(&sizeopt_data, sizeopt_callback_log, 
					sizeopt_outfile);
		}

		/* fill size optimization data */

		mpz_set(sizeopt_data.gmp_N, n);
		sizeopt_data.degree = degree;
		sizeopt_data.max_stage1_norm = params->stage1_norm;
		sizeopt_data.max_sizeopt_norm = params->stage2_norm;

		if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) &&
		    gpu_sizeopt_wanted(degree)) {
			stage1_data.gpu_sizeopt = 1;
			stage1_data.sizeopt_norm = params->stage2_norm;
			stage1_data.sizeopt_callback =
					stage1_sizeopt_callback;
		}
	}

	/* set up root optimization */

	if (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) {

		poly_rootopt_init(&rootopt_data, obj, rootopt_callback, 
				&rootopt_callback_data);

		/* fill root optimization data */

		mpz_set(rootopt_data.gmp_N, n);
		rootopt_data.degree = degree;
		rootopt_data.max_sizeopt_norm = params->stage2_norm;
		/* polynomials are kept down to a quarter of the minimum E
		   (MSIEVE_POLY_KEEP_E, 1 = msieve's): the best one found is
		   then used even when none reaches the minimum, instead of a
		   second search (a c90's took 140 s) */
		double keep = 0.25;
		{
			const char *e = getenv("MSIEVE_POLY_KEEP_E");

			if (e != NULL)
				keep = atof(e);
			if (!(keep > 0 && keep < 1))
				keep = 1;
		}
		rootopt_data.min_e = params->final_norm * keep;
		rootopt_data.min_e_bernstein = 0;

		/* smaller problems (especially degree 5) run much faster 
		   when Bernstein's scoring function is used to weed out 
		   polynomials that probably cannot get their Murphy score 
		   optimized enough to exceed the E-value bound. Degree 4's
		   bound stays msieve's (it passes Murphy scores down to
		   1/30000 of the minimum); degree 5's is derived from the
		   lower minimum too: msieve's passed nothing in a c110's
		   whole first round (1/6.6 of the minimum) */

		if (degree == 4) {
			rootopt_data.min_e_bernstein = params->final_norm / 
					(30000 * pow(2.5, (params->digits - 90) / 5));
		}
		else if (degree == 5) {
			rootopt_data.min_e_bernstein = params->final_norm *
					keep /
					(2.2 * pow(3.0, (params->digits - 100) / 10));
		}

		poly_out_name(buf, obj, "p");
		rootopt_callback_data.config = config;
		rootopt_callback_data.report_min_e = (poly_report_norm > 0) ?
				poly_report_norm : params->final_norm;
		rootopt_callback_data.all_poly_file = fopen(buf, "a");
		if (rootopt_callback_data.all_poly_file == NULL) {
			printf("error: cannot open root opt file\n");
			exit(-1);
		}
		rootopt_start(obj, &rootopt_data);
	}


	/* the stage 2 workers (worker 0 is sizeopt_data / rootopt_data) */

	if ((obj->flags & (MSIEVE_FLAG_NFS_POLYSIZE |
			   MSIEVE_FLAG_NFS_POLYROOT)) && num_stage2 > 1) {
		bool root = (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) ? true : false;
		bool size = (obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) ? true : false;

		stage2_workers_init(&workers, num_stage2, obj, n,
				size ? &sizeopt_data : NULL,
				root ? &rootopt_data : NULL,
				root ? sizeopt_callback : sizeopt_callback_log,
				sizeopt_outfile, &rootopt_callback_data);
		logprintf(obj, "stage 2 threads: %u\n", num_stage2);

		if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) && size) {
			stage1_data.num_stage2_threads = num_stage2;
			stage1_data.stage2_callback_data =
						(void **)workers.sizeopt;
		}
		else {
			thread_control_t tc = {0};

			stage2_pool = threadpool_init(num_stage2, 1000, &tc);
		}
	}

	if (obj->flags & MSIEVE_FLAG_NFS_POLY1) {

		poly_stage1_run(obj, &stage1_data);

		poly_stage1_free(&stage1_data);
		if (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) {
			rootopt_finish();
			fclose(rootopt_callback_data.all_poly_file);
			poly_sizeopt_free(&sizeopt_data);
			poly_rootopt_free(&rootopt_data);
		}
		else if (obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) {
			fclose(sizeopt_outfile);
			poly_sizeopt_free(&sizeopt_data);
		}
	}
	else if (obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) {
		mpz_t ad, p, m;

		mpz_init(ad);
		mpz_init(p);
		mpz_init(m);

		sprintf(buf, "%s.m", obj->savefile.name);
		stage1_outfile = fopen(buf, "r");
		if (stage1_outfile == NULL) {
			printf("error: cannot open sizeopt input file\n");
			exit(-1);
		}

		while (1) {
			if (fgets(buf, sizeof(buf), stage1_outfile) == NULL)
				break;

			if (gmp_sscanf(buf, "%Zd %Zd %Zd", ad, p, m) != 3)
				continue;

			/* MPI: line k is process (k mod processes)'s */
			if ((line_index++ % poly_nranks) != poly_rank)
				continue;

			if (stage2_pool != NULL) {
				stage2_line_t *l = stage2_line_new(&workers, false);

				mpz_set(l->c[0], ad);
				mpz_set(l->c[1], p);
				mpz_set(l->c[2], m);
				stage2_line_submit(stage2_pool, l);
			}
			else {
				poly_sizeopt_run(&sizeopt_data, ad, p, m);
			}
			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
				break;
		}
		if (stage2_pool != NULL) {
			if (!(obj->flags & MSIEVE_FLAG_STOP_SIEVING))
				threadpool_drain(stage2_pool, 1);
			threadpool_free(stage2_pool);
			stage2_pool = NULL;
		}

		mpz_clear(ad);
		mpz_clear(p);
		mpz_clear(m);
		fclose(stage1_outfile);
		poly_sizeopt_free(&sizeopt_data);
		if (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) {
			rootopt_finish();
			fclose(rootopt_callback_data.all_poly_file);
			poly_rootopt_free(&rootopt_data);
		}
	}
	else if (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) {
		mpz_t alg_coeffs[MAX_POLY_DEGREE + 1];
		mpz_t rat_coeffs[2];

		for (uint32_t i = 0; i <= MAX_POLY_DEGREE; i++)
			mpz_init(alg_coeffs[i]);
		for (uint32_t i = 0; i <= 1; i++)
			mpz_init(rat_coeffs[i]);

		sprintf(buf, "%s.ms", obj->savefile.name);
		sizeopt_outfile = fopen(buf, "r");
		if (sizeopt_outfile == NULL) {
			printf("error: cannot open rootopt input file\n");
			exit(-1);
		}

		while (1) {
			int c;
			char *tmp = buf;

			/* this reading scheme is unintuitive but more
			   resistant to corrupted input lines */

			if (fgets(tmp, sizeof(buf), sizeopt_outfile) == NULL)
				break;

			for (uint32_t i = 0; i <= degree; i++) {
				if (gmp_sscanf(tmp, "%Zd%n", 
					    alg_coeffs[degree - i], &c) != 1)
						break;
				tmp += c;
			}

			fputs(tmp, stdout);

			if (gmp_sscanf(tmp, "%Zd %Zd", 
					rat_coeffs[1], rat_coeffs[0]) != 2)
				continue;

			/* MPI: line k is process (k mod processes)'s */
			if ((line_index++ % poly_nranks) != poly_rank)
				continue;

			/* make later code figure out the initial poly score */

			if (stage2_pool != NULL) {
				stage2_line_t *l = stage2_line_new(&workers, true);

				for (uint32_t i = 0; i <= degree; i++)
					mpz_set(l->c[i], alg_coeffs[i]);
				mpz_set(l->r[0], rat_coeffs[0]);
				mpz_set(l->r[1], rat_coeffs[1]);
				stage2_line_submit(stage2_pool, l);
			}
			else {
				rootopt_run(&rootopt_data, alg_coeffs,
						rat_coeffs, 0, 0);
			}
			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
				break;
		}
		if (stage2_pool != NULL) {
			if (!(obj->flags & MSIEVE_FLAG_STOP_SIEVING))
				threadpool_drain(stage2_pool, 1);
			threadpool_free(stage2_pool);
			stage2_pool = NULL;
		}

		for (uint32_t i = 0; i <= MAX_POLY_DEGREE; i++)
			mpz_clear(alg_coeffs[i]);
		for (uint32_t i = 0; i <= 1; i++)
			mpz_clear(rat_coeffs[i]);
		fclose(sizeopt_outfile);
		rootopt_finish();
			fclose(rootopt_callback_data.all_poly_file);
		poly_rootopt_free(&rootopt_data);
	}

	if (workers.num > 1)
		stage2_workers_free(&workers);
	mutex_free(&stage2_out_lock);
	if (polysel_skipped > 0 || rootopt_dropped > 0)
		logprintf(obj, "time limit reached: %" PRIu64 " size-optimized "
				"polynomials not root-optimized\n",
				polysel_skipped + rootopt_dropped);
	if (topk_k > 0) {
		logprintf(obj, "root optimization of the %u best by size so "
				"far: %" PRIu64 " size-optimized polynomials "
				"not root-optimized\n", topk_k, topk_skipped);
		mutex_free(&topk_lock);
		free(topk_best);
		topk_best = NULL;
		topk_k = 0;
	}
	polysel_wall_end = 0;
	rootopt_wall_end = 0;
	rootopt_dropped = 0;
	polysel_skipped = 0;

#ifdef HAVE_MPI
	if (poly_nranks > 1) {
		if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) &&
		    !(obj->flags & MSIEVE_FLAG_NFS_POLYSIZE))
			poly_mpi_merge_file(obj, "m");
		if ((obj->flags & MSIEVE_FLAG_NFS_POLYSIZE) &&
		    !(obj->flags & MSIEVE_FLAG_NFS_POLYROOT))
			poly_mpi_merge_file(obj, "ms");
		if (obj->flags & MSIEVE_FLAG_NFS_POLYROOT) {
			poly_mpi_merge_file(obj, "p");
			poly_mpi_merge_best(obj, config, degree);
		}
	}
#endif
}
