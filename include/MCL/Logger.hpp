// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_LOGGER_HPP
#define MCL_LOGGER_HPP 1

#include "Singleton.hpp"
#include <Eigen/Core>
#include <unordered_map>
#include <vector>
#include "MCL/MicroTimer.hpp"

namespace mcl
{
	class SetCounterCurrentFrame;
	class FunctionTimerWrapper;
	class LogFrameWrapper;
}

// Set counters/values/runtime for the current frame
// Use:
//   mclSetCounter("label1", int_counter);
//   mclSetValue("label2", dbl_value);
//   mclAddRuntime(elapsed_s);
//   mclAddFrameRuntime(iter, elapsed_s); // special: adds time to specific iter
//
#define	mclSetCounter(l, cnt) mcl::SetCounterCurrentFrame::set_int(l, cnt)
#define	mclSetValue(l, cnt) mcl::SetCounterCurrentFrame::set_double(l, cnt)
#define	mclAddRuntime(cnt) mcl::SetCounterCurrentFrame::add_runtime(cnt)
#define	mclAddFrameRuntime(iter, cnt) mcl::SetCounterCurrentFrame::add_runtime(iter, cnt)

// Timing a scope
// Use:
//   { // region you want to time
//     mclStart("MyClass::MyFunction");
//     ...
//   } // timer stops
//
#define	mclStart(f) mcl::FunctionTimerWrapper ftWrapper_(f)

// Special case: incrementing a frame
// Use:
//   MySolver::iterate()
//   {
//     mclStartFrame();
//     ...
//   }
// Which will increment Logger::frame_number and call
// Logger::begin_frame(). When function ends, calls Logger::end_frame().
#define	mclStartFrame() mcl::LogFrameWrapper lfWrapper_

namespace mcl
{

class PerFrameData
{
public:
	PerFrameData();

	// Inserts 0 if the counter/timing doesn't exist.
	int& get_int_counter(const std::string& l);
	double& get_double_counter(const std::string& l);
	double get_time_ms(const std::string &l) const;

	// Starts a timer with the current label.
	void start_timer(const std::string &l);

	// Stops a timer with the current label.
	// If timings already exist for this label, they are summed.
	void stop_timer(const std::string &l);

	// Data exposed to make it easier to read.
	std::unordered_map<std::string, int> int_counters;
	std::unordered_map<std::string, double> double_counters;
	std::unordered_map<std::string, double> timings_ms; // stopped timers
	std::unordered_map<std::string, MicroTimer> timers; // running timers

	template<class Archive> void serialize(Archive& archive);
};

class Logger : public Singleton<Logger>
{
public:
	Logger();
	~Logger() {}

	int curr_frame;
	bool print_timer_start; // for debugging
	bool pause_simulation; // for debugging

	// Clears all logged data
	void clear();
	
	// Returns directory of logger output
	static std::string get_output_dir();

	// Custom logging callback
	std::function<void()> begin_frame;
	std::function<void()> end_frame;

	// Functions that are called at the beginning and ending
	// of a function. Used for timing and debugging.
	void start(int iter, const std::string &f);
	void stop(int iter, const std::string &f);

	// Variable used for keeping track of the per-frame runtime.
	void add_runtime_s(int iter, double s);

	// Labeled counters keep track of per-iteration variables.
	// For get counter, if iter < -1, returns sum.
	void set_int_counter(int iter, const std::string &l, const int &v);
	void add_to_int_counter(int iter, const std::string &l, const int &v);
	int get_int_counter(int iter, const std::string &l);
	bool has_int_counter(int iter, const std::string &l) const; // true if counter exists at iteration
	void set_double_counter(int iter, const std::string &l, const double &v);
	void add_to_double_counter(int iter, const std::string &l, const double &v);
	bool has_double_counter(int iter, const std::string &l) const; // true if counter exists at iteration
	double get_double_counter(int iter, const std::string &l);

	// Prints aggregate list
	std::string print_timers();
	std::string print_counters();

	// Writes timings/counters/etc to file
	static std::string make_write_prefix(std::string test);
	void write_csv(std::string prefix); // per-frame data
	void write_timings(std::string prefix); // summary
	void write_counters(std::string prefix); // summary
	void write_all(std::string test)
	{
		std::string pf = make_write_prefix(test);
		write_csv(pf);
		write_timings(pf);
		write_counters(pf);
	}

	template<class Archive> void serialize(Archive& archive);

protected:
	std::unordered_map<int,double> per_frame_runtime_s; // per frame runtime
	std::unordered_map<int,PerFrameData> per_frame_data;

	// Returns data associated with this iteration.
	// Must be called once in main thread for data to be created.
	PerFrameData& get_frame_data(int iter);

};

class SetCounterCurrentFrame
{
public:
	static void set_int(const char *l, int cnt);
	static void set_double(const char *l, double cnt);
	static void add_runtime(double cnt);
	static void add_runtime(int iter, double cnt);
};

class FunctionTimerWrapper
{
public:
	const std::string f;
	FunctionTimerWrapper(const char *f_);
	~FunctionTimerWrapper();
};

class LogFrameWrapper
{
public:
	MicroTimer timer;
	LogFrameWrapper();
	~LogFrameWrapper();
};

} // ns mcl

#endif
