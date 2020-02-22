#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include<doctest.h>

#include<SparseDNN/utility/thread_pool.hpp>
#include<thread>

TEST_CASE("sum"){
  size_t num_threads = std::thread::hardware_concurrency();
  for(size_t i=1; i<num_threads; ++i){
    ThreadPool thread_pool{i};
    std::vector<std::future<int> > futures;
    int sum{0};
    for(int j=1; j<=5; ++j){
      futures.push_back(thread_pool.enqueue([j](){return (2*j-1)+(2*j);}));
    }
    for(auto& result:futures){
      result.wait();
    }

    for(auto& result:futures){
      sum += result.get();
    }
    CHECK(sum==55);
  }
}

TEST_CASE("create pool" * doctest::timeout(300)){
  size_t num_threads = std::thread::hardware_concurrency();
  for(size_t i=1; i<num_threads; ++i){
    ThreadPool t(i);
  }
}

TEST_CASE("enqueue type" * doctest::timeout(300)){
  auto f = std::make_shared<std::function<void()> >;

  ThreadPool t(1);
  t.enqueue([]{});
  t.enqueue(std::function<void()>([](){}));
  t.enqueue(*f);
}

TEST_CASE("enqueue large size" * doctest::timeout(300)){
  ThreadPool t(1);
  for(size_t i=0; i<65536; ++i){
    t.enqueue([]{});
  }
}
