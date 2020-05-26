

namespace snig {

//TODO:

template <typename T>
class Base {

  protected:

    // Add a virtual destructor
    virtual ~Base();
  
    // TODO: add a log method
    //  API: cout("my ", string, " is ", a, b, '\n');
    //       -> cout << "my" << string << " is " << a << b << '\n';
    template <typename... ArgsT>
    void log(ArgsT&&... args) {
      _cout(std::forward<ArgsT>(args)...);  // search for "universal/perfect forwarding"
    }

    // TODO: you may add a method to compute timer 

    // TODO: consider parameterizing the kernel thread configuration
    
  private:

    template <typename T>
    void _cout(T&& last) {
      std::cout << last; // something like this
    }

    template <typename First, typename... Remain>
    void _cout(First&& item, Remain&&... remain) {
      std::cout << item;
      _cout(std::forward<Remain>(remain)...);
    }


};



}  // end of namespace snig
