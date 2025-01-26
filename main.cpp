#include <string.h>
#include <stdio.h>
#include <vector>
#include <atomic>
#include <thread>
#include <immintrin.h>
#include <time.h>
#include <unistd.h>

#define assert(x) do { if (!(x)) { fprintf(stderr, "***ASSERT FAILED*** %s\n", #x); __builtin_trap(); } } while (0)

#define DBG 0

#if DBG
#define print(...) printf(__VA_ARGS__)
#else
#define print(...)
#endif

namespace {

#define is_pow2(x) ((x & (x - 1)) == 0)

constexpr size_t MAX_WORK = 2048;
static_assert(is_pow2(MAX_WORK));

#define WORK_SIG(name) void name(void *arg)
typedef WORK_SIG(WorkFn);

struct Work {
  std::atomic<bool> *is_done;
  void *arg;
  WorkFn *fn;
};

struct WorkCtl {
  Work work[MAX_WORK];
  std::vector<std::thread> threads;
  std::atomic<size_t> head; // for manager
  std::atomic<size_t> tail; // for workers
  std::atomic<bool> die;
};

struct WorkerInit {
  int id;
  WorkCtl *ctl;
};

int worker_loop(WorkerInit init)
{
  WorkCtl *ctl = init.ctl;
  [[maybe_unused]] int id = init.id;

  size_t lc = 0;
  size_t wc = 0;
  int mask = 1;
  int max = 64;
  while (ctl->die.load(std::memory_order_acquire) == false) {
    size_t tail = ctl->tail.load(std::memory_order_acquire);
    size_t head = ctl->head.load(std::memory_order_acquire);

    // I am not going to count this as increasing the loop count because we are not contending for
    // work here, just checking if any has become available.  Maybe 'loop_count' should be renamed
    // to indicate that is more so a measure of contention, as opposed to a measure of iterations.
    if (tail == head) {
      sleep(0);
      continue;
    }

    Work w = ctl->work[tail & (MAX_WORK-1)];
    if (std::atomic_compare_exchange_strong(&ctl->tail, &tail, tail + 1) == true) {
      print("Worker %i got work: head = %lu, tail = %lu (loop_count = %lu)\n", id, head, tail, loop_count);

      w.fn(w.arg);
      if (w.is_done != nullptr)
        w.is_done->store(true, std::memory_order_release);
      print("Worker %i completed work: head = %lu, tail = %lu (loop_count = %lu)\n", id, head, tail, loop_count);

      wc += 1;
      mask = 1;
    }

    for (int i=0; i < mask; ++i)
      _mm_pause();
    mask = mask < max ? max << 1 : max;

    lc += 1;
  }

  printf("Worker %i returning, completed %lu loops and %lu work\n", id, lc, wc);
  return 0;
}

void init_workctl(WorkCtl &ctl)
{
  int num = std::thread::hardware_concurrency();
  print("Have %i threads\n", num);

  ctl.head = 0;
  ctl.tail = 0;
  ctl.die = false;

  for (int i=0; i < num; ++i) {
    WorkerInit a;
    a.id = i;
    a.ctl = &ctl;
    ctl.threads.emplace_back(worker_loop, a);
  }
}

int add_work(WorkCtl& ctl, Work *work, int count)
{
  int pos = 0;
  size_t head = ctl.head;
  size_t tail = ctl.tail.load(std::memory_order_acquire);
  while (count - pos > 0 && head - tail < MAX_WORK) {
    size_t dist = MAX_WORK - (head - tail);
    size_t rem = count - pos;

    if (rem < dist)
      dist = rem;

    for (size_t i=0; i < dist; ++i)
      ctl.work[(head + i) & (MAX_WORK-1)] = work[pos + i];

    pos += dist;
    head += dist;

    ctl.head.store(head, std::memory_order_release);
    tail = ctl.tail.load(std::memory_order_acquire);
  }
  return pos;
}

} // namespace

struct WorkTest {
  int *from;
  int *to;
  int cnt;
  int idx;
};

WORK_SIG(work_test)
{
  WorkTest w = *(WorkTest *)arg;

  print("Doing work idx %i (to = %p, from = %p)\n", w.idx, (void*) w.to, (void*) w.from);
  for (int i = 0; i < w.cnt; ++i) {
    if (i < 5) {
      print("%i + %i\n", w.to[i], w.from[i]);
    }
    w.to[i] += w.from[i];
  }
  print("\n");
}

int main() {

  int cnt = 100'000'000;
  std::vector<int> to(cnt);
  std::vector<int> from(cnt);

  for (int i=0; i < cnt; ++i)
    from[i] = i + 1;

  WorkCtl ctl;
  init_workctl(ctl);

  int stride = 50;
  std::vector<WorkTest> wt(cnt / stride);
  std::vector<Work> w(cnt / stride);
  std::vector<std::atomic<bool>> b(cnt / stride);

  for (int i=0; i < cnt / stride; ++i) {
    wt[i].cnt = stride;
    wt[i].to = to.data() + i * stride;
    wt[i].from = from.data() + i * stride;
    wt[i].idx = i;

    w[i].arg = (void*) &wt[i];
    w[i].fn = work_test;
    w[i].is_done = &b[i];
  }

  std::clock_t tim = std::clock();

  for (int i=0; i < cnt / stride;) {
    i += add_work(ctl, w.data() + i, cnt / stride - i);
  }

  unsigned short s = 0;
  while(s < 0xffff)
    for (int i=0; i < 16; ++i)
      s |= b[b.size()-1].load(std::memory_order_acquire) << i;

  _mm_mfence();
  printf("Elapsed %f seconds\n", (double) (clock() - tim) / CLOCKS_PER_SEC);

  for (size_t i = 0; i < to.size() - 1; ++i) {
    if (to[i] != to[i+1] - 1) {
      printf("%lu: %i, %lu: %i\n", i, to[i], i+1, to[i+1]);
    }
    assert(to[i] == to[i+1] - 1);
  }

  ctl.die.store(true, std::memory_order_release);
  for (std::thread &t : ctl.threads)
    t.join();

  return 0;
}
