// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#include "Logger.hpp"
#include "MCL/AssertHandler.hpp"

#include <sstream>
#include <fstream>
#include <iomanip> // setprecision
#include <set>
#include <experimental/filesystem>
#include <mutex>

#include <cereal/cereal.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>

namespace mcl
{

// I'm just going to wrap everything in a mutex to be safe.
static std::mutex log_mutex;

Logger::Logger()
{
	clear();
}

void Logger::clear()
{
	curr_frame = -1;
	print_timer_start = false;
	pause_simulation = false;
	per_frame_runtime_s.clear();
	per_frame_runtime_s.emplace(-1, 0);
	per_frame_data.clear();
}

void Logger::start(int iter, const std::string &f)
{
	const std::lock_guard<std::mutex> lock(log_mutex);
	PerFrameData &data = get_frame_data(iter);
	data.start_timer(f);
	if (print_timer_start) {
		printf("Starting %s\n", f.c_str());
	}
}

void Logger::stop(int iter, const std::string &f)
{
	const std::lock_guard<std::mutex> lock(log_mutex);
	PerFrameData &data = get_frame_data(iter);
	data.stop_timer(f);
	if (print_timer_start) {
		printf("Stopping %s\n", f.c_str());
	}
}


void Logger::add_runtime_s(int iter, double s)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	std::unordered_map<int,double>::iterator it = per_frame_runtime_s.emplace(iter, 0).first;
	it->second += s;
}

void Logger::set_int_counter(int iter, const std::string &l, const int &v)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	if (print_timer_start) {
		printf("%s: %d\n", l.c_str(), v);
	}
	PerFrameData &data = get_frame_data(iter);
	data.get_int_counter(l) = v;
}

void Logger::add_to_int_counter(int iter, const std::string &l, const int &v)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	PerFrameData &data = get_frame_data(iter);
	data.get_int_counter(l) += v;
}

int Logger::get_int_counter(int iter, const std::string &l)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	PerFrameData &data = get_frame_data(iter);
	return data.get_int_counter(l);
}

bool Logger::has_int_counter(int iter, const std::string &l) const
{
	std::unordered_map<int,PerFrameData>::const_iterator it = per_frame_data.find(iter);
	if (it == per_frame_data.end())
		return false;

	return it->second.int_counters.count(l) > 0;
}

void Logger::set_double_counter(int iter, const std::string &l, const double &v)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	if (print_timer_start) {
		printf("%s: %f\n", l.c_str(), v);
	}
	PerFrameData &data = get_frame_data(iter);
	data.get_double_counter(l) = v;
}

void Logger::add_to_double_counter(int iter, const std::string &l, const double &v)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	PerFrameData &data = get_frame_data(iter);
	data.get_double_counter(l) += v;
}

double Logger::get_double_counter(int iter, const std::string &l)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	PerFrameData &data = get_frame_data(iter);
	return data.get_double_counter(l);
}

bool Logger::has_double_counter(int iter, const std::string &l) const
{
	std::unordered_map<int,PerFrameData>::const_iterator it = per_frame_data.find(iter);
	if (it == per_frame_data.end())
		return false;

	return it->second.double_counters.count(l) > 0;
}

std::string Logger::print_timers()
{
	// Gather a list of all timer labels throughout the simulation.
	// Some labels/timings may not occur every iteration.
	std::set<std::string> timer_keys;
	std::unordered_map<std::string, int> count;
	{
		std::unordered_map<int,PerFrameData>::const_iterator it = per_frame_data.begin();
		for (; it != per_frame_data.end(); ++it)
		{
			std::set<std::string> keys_this_iter;
			std::unordered_map<std::string, double>::const_iterator it2 = it->second.timings_ms.begin();
			for (; it2 != it->second.timings_ms.end(); ++it2)
			{
				timer_keys.emplace(it2->first);
				keys_this_iter.emplace(it2->first);
			}

			// Append per-iter count
			for (std::set<std::string>::iterator it3 = keys_this_iter.begin(); it3 != keys_this_iter.end(); ++it3)
			{
				if (count.count(*it3)==0) { count[*it3] = 1; }
				else { count[*it3] += 1; }
			}
		}
	}

	if (timer_keys.size()==0)
		return "";

	std::unordered_map<std::string, double> total_ms;
	std::unordered_map<std::string, double> max_ms;

	// Loop each iteration that we logged.
	int iter = -1;
	while (true)
	{	
		if (iter >= 0 && per_frame_data.count(iter)==0)
			break;

//		mclAssert(runtime_s.count(iter)>0, "Logger::write_csv: print_timers not set");
//		double curr_runtime_s = runtime_s.find(iter)->second;

		// Otherwise this function inserts if not exists,
		// which (in this cases) we don't want to do.
		PerFrameData& data = get_frame_data(iter);

		for (std::set<std::string>::iterator it = timer_keys.begin();
			it != timer_keys.end(); ++it)
		{
			double time_ms = data.get_time_ms(*it);

			if (total_ms.count(*it)) { total_ms[*it] += time_ms; }
			else { total_ms[*it] = time_ms; }

			if (max_ms.count(*it))
			{
				double prev = max_ms[*it];
				max_ms[*it] = std::max(time_ms, prev);
			}
			else { max_ms[*it] = time_ms; }
		}

		iter++;
	}

	// For each timing, sort
	std::vector<std::pair<double, std::string> > sorted_total;
	std::vector<std::pair<double, std::string> > sorted_max;
	std::vector<std::pair<double, std::string> > sorted_avg;

	for (std::set<std::string>::iterator it = timer_keys.begin(); it != timer_keys.end(); ++it)
	{
		std::unordered_map<std::string, double>::const_iterator it_tot = total_ms.find(*it);
		mclAssert(it_tot != total_ms.end());
		std::unordered_map<std::string, double>::const_iterator it_max = max_ms.find(*it);
		mclAssert(it_max != max_ms.end());
		std::unordered_map<std::string, int>::const_iterator it_count = count.find(*it);
		mclAssert(it_count != count.end() && it_count->second > 0);

		sorted_total.emplace_back(it_tot->second, it_tot->first);
		sorted_max.emplace_back(it_max->second, it_max->first);
		sorted_avg.emplace_back(it_tot->second / double(it_count->second), it_tot->first);
	}

	std::sort(sorted_total.begin(), sorted_total.end());
	std::reverse(sorted_total.begin(), sorted_total.end());
	std::sort(sorted_max.begin(), sorted_max.end());
	std::reverse(sorted_max.begin(), sorted_max.end());
	std::sort(sorted_avg.begin(), sorted_avg.end());
	std::reverse(sorted_avg.begin(), sorted_avg.end());

	std::stringstream ss;
	int nt = timer_keys.size();
	for (int i=0; i<nt; ++i)
	{
		const std::pair<double,std::string> &p = sorted_total[i];
		ss << "Total time " << p.second << ": " << p.first << " ms" << std::endl;
	}
	for (int i=0; i<nt; ++i)
	{
		const std::pair<double,std::string> &p = sorted_max[i];
		ss << "Max time " << p.second << ": " << p.first << " ms" << std::endl;
	}
	for (int i=0; i<nt; ++i)
	{
		const std::pair<double,std::string> &p = sorted_avg[i];
		ss << "Avg time (per iter) " << p.second << ": " << p.first << " ms" << std::endl;
	}

	return ss.str();
}

std::string Logger::print_counters()
{
	std::unordered_map<std::string, int> int_total_count;
	std::unordered_map<std::string, int> int_max_count;
	std::unordered_map<std::string, int> int_count;
	std::unordered_map<std::string, double> double_total_count;
	std::unordered_map<std::string, double> double_max_count;
	std::unordered_map<std::string, int> double_count;
	int tot_data = 0;

	// Some iterations might not have a specific label
	for (std::unordered_map<int,PerFrameData>::const_iterator it = per_frame_data.begin();
		it != per_frame_data.end(); ++it)
	{
		{ // int counters
			std::unordered_map<std::string, int>::const_iterator it2 = it->second.int_counters.begin();
			for (; it2 != it->second.int_counters.end(); ++it2)
			{
				tot_data++;
				const std::string &l = it2->first;
				int cnt = it2->second;

				if (int_total_count.count(l)) { int_total_count[l] += cnt; }
				else { int_total_count[l] = cnt; }

				if (int_max_count.count(l)) {
					int prev = int_max_count[l];
					int_max_count[l]=std::max(prev, cnt);
				} else { int_max_count[l] = cnt; }

				if (int_count.count(l)==0) { int_count[l] = 1; }
				else { int_count[l] += 1; }
			}
		}
		{ // dbl counters
			std::unordered_map<std::string, double>::const_iterator it2 = it->second.double_counters.begin();
			for (; it2 != it->second.double_counters.end(); ++it2)
			{
				tot_data++;
				const std::string &l = it2->first;
				double cnt = it2->second;

				if (double_total_count.count(l)) { double_total_count[l] += cnt; }
				else { double_total_count[l] = cnt; }

				if (double_max_count.count(l)) {
					double prev = double_max_count[l];
					double_max_count[l]=std::max(prev, cnt);
				} else { double_max_count[l] = cnt; }

				if (double_count.count(l)==0) { double_count[l] = 1; }
				else { double_count[l] += 1; }
			}
		}
	} // end loop per iteration data

	if (tot_data==0)
		return "";

	// For each counter, sort
	std::vector<std::pair<int, std::string> > int_sorted_total;
	std::vector<std::pair<int, std::string> > int_sorted_max;
	std::vector<std::pair<double, std::string> > int_sorted_avg;
	std::vector<std::pair<double, std::string> > double_sorted_total;
	std::vector<std::pair<double, std::string> > double_sorted_max;
	std::vector<std::pair<double, std::string> > double_sorted_avg;

	for (std::pair<std::string, int> it : int_total_count) { int_sorted_total.emplace_back(it.second, it.first); }
	for (std::pair<std::string, int> it : int_max_count) { int_sorted_max.emplace_back(it.second, it.first); }
	for (std::pair<std::string, double> it : int_total_count) { int_sorted_avg.emplace_back(it.second/double(int_count[it.first]), it.first); }
	for (std::pair<std::string, double> it : double_total_count) { double_sorted_total.emplace_back(it.second, it.first); }
	for (std::pair<std::string, double> it : double_max_count) { double_sorted_max.emplace_back(it.second, it.first); }
	for (std::pair<std::string, double> it : double_total_count) { double_sorted_avg.emplace_back(it.second/double(double_count[it.first]), it.first); }

	std::sort(int_sorted_total.begin(), int_sorted_total.end());
	std::reverse(int_sorted_total.begin(), int_sorted_total.end());
	std::sort(int_sorted_max.begin(), int_sorted_max.end());
	std::reverse(int_sorted_max.begin(), int_sorted_max.end());
	std::sort(int_sorted_avg.begin(), int_sorted_avg.end());
	std::reverse(int_sorted_avg.begin(), int_sorted_avg.end());

	std::sort(double_sorted_total.begin(), double_sorted_total.end());
	std::reverse(double_sorted_total.begin(), double_sorted_total.end());
	std::sort(double_sorted_max.begin(), double_sorted_max.end());
	std::reverse(double_sorted_max.begin(), double_sorted_max.end());
	std::sort(double_sorted_avg.begin(), double_sorted_avg.end());
	std::reverse(double_sorted_avg.begin(), double_sorted_avg.end());

	bool print_dbl_counters = true;

	std::stringstream ss;
	for (std::pair<int, std::string> it : int_sorted_total)
	{
		ss << "Total count " << it.second << ": " << it.first << std::endl;
	}
	for (std::pair<int, std::string> it : int_sorted_max)
	{
		ss << "Max count " << it.second << ": " << it.first << std::endl;
	}
	for (std::pair<double, std::string> it : int_sorted_avg)
	{
		ss << "Avg count (per iter) " << it.second << ": " << it.first << std::endl;
	}
	if (print_dbl_counters)
	{
		for (std::pair<int, std::string> it : double_sorted_total)
		{
			ss << "Total count " << it.second << ": " << it.first << std::endl;
		}
		for (std::pair<double, std::string> it : double_sorted_max)
		{
			ss << "Max count " << it.second << ": " << it.first << std::endl;
		}
		for (std::pair<double, std::string> it : double_sorted_avg)
		{
			ss << "Avg count (per iter) " << it.second << ": " << it.first << std::endl;
		}
	}

	return ss.str();
}

std::string Logger::make_write_prefix(std::string test)
{
	return std::string(MCL_APP_OUTPUT_DIR)+'/'+test+'/'+test+'_';
}

void Logger::write_csv(std::string prefix)
{
	if (per_frame_data.size()==0 && per_frame_runtime_s.size()==0)
		return;

	std::string fn = prefix + "log.csv";

	// Make directories as needed
	std::experimental::filesystem::path p(fn);
	std::experimental::filesystem::create_directories(p.parent_path());

	if (print_timer_start) {
		printf("Saving log file: %s\n", fn.c_str());
	}

	// Make sure a per-frame-data exists for each runtime_s
	// To make looping/exporting easier
	for (std::unordered_map<int,double>::iterator it = per_frame_runtime_s.begin();
		it != per_frame_runtime_s.end(); ++it)
		get_frame_data(it->first);

	// Get a list of keys for all timers and counters
	// These are used as the columns of CSV file.
	// Some data might exist in some iters that it might not exist in others.
	// We have to loop ALL data to get the keys.
	std::set<std::string> int_counter_keys;
	std::set<std::string> double_counter_keys;
	std::set<std::string> timer_keys;
	{
		std::unordered_map<int,PerFrameData>::const_iterator it = per_frame_data.begin();
		for (; it != per_frame_data.end(); ++it)
		{
			{
				std::unordered_map<std::string, int>::const_iterator it2 = it->second.int_counters.begin();
				for (; it2 != it->second.int_counters.end(); ++it2) { int_counter_keys.emplace(it2->first); }
			}
			{
				std::unordered_map<std::string, double>::const_iterator it2 = it->second.double_counters.begin();
				for (; it2 != it->second.double_counters.end(); ++it2) { double_counter_keys.emplace(it2->first); }
			}
			{
				std::unordered_map<std::string, double>::const_iterator it2 = it->second.timings_ms.begin();
				for (; it2 != it->second.timings_ms.end(); ++it2) { timer_keys.emplace(it2->first); }
			}
		}
	}

	std::ofstream ofout(fn.c_str());
	mclAssert(ofout.good(), "Logger::write_csv: failed to create file");

	//
	// Loop each iteration that we logged.
	// For each iteration, loop all keys and write data
	//
	double cumulative_runtime_s = 0;
	int iter = -2; // -2 => write CSV header
	while (true)
	{	
		// When we no longer have data, break
		std::stringstream line;

		// Write header
		if (iter==-2)
		{
			line << "iter,runtime_s,"; // these are always first

			for (std::set<std::string>::iterator it = int_counter_keys.begin();
				it != int_counter_keys.end(); ++it) { line << *it << ','; }

			for (std::set<std::string>::iterator it = double_counter_keys.begin();
				it != double_counter_keys.end(); ++it) { line << *it << ','; }

			for (std::set<std::string>::iterator it = timer_keys.begin();
				it != timer_keys.end(); ++it) { line << *it << ','; }

			ofout << line.str() << std::endl;
			iter++;
			continue;
		}

		// Stop writing as soon as we encounter an iteration with no data
		if (iter >= 0 && per_frame_data.count(iter)==0)
			break;

		// If it's iter=-1 and we don't have data, then we never started
		// a simulation. Just quit.
		if (iter==-1 && per_frame_runtime_s.count(iter)==0)
			break;

		// Otherwise, we should have set runtime_s for every iteration!
		mclAssert(per_frame_runtime_s.count(iter)>0, "Logger::write_csv: runtime_s not set");
		double curr_runtime_s = per_frame_runtime_s.find(iter)->second;
		cumulative_runtime_s += curr_runtime_s;

		// Otherwise this function inserts if not exists,
		// which (in this cases) we don't want to do.
		PerFrameData& data = get_frame_data(iter);

		line << iter << ",";
		line << std::setprecision(12);
		line << cumulative_runtime_s << ",";

		for (std::set<std::string>::iterator it = int_counter_keys.begin();
			it != int_counter_keys.end(); ++it)
		{
			int v = data.get_int_counter(*it);
			line << v << ",";
		}

		for (std::set<std::string>::iterator it = double_counter_keys.begin();
			it != double_counter_keys.end(); ++it)
		{
			double d = data.get_double_counter(*it);
			line << d << ",";
		}

		for (std::set<std::string>::iterator it = timer_keys.begin();
			it != timer_keys.end(); ++it)
		{
			double time_ms = data.get_time_ms(*it);
			line << time_ms << ",";
		}

		ofout << line.str() << std::endl;
		iter++;
	}

	ofout.close();

} // end write csv

void Logger::write_timings(std::string prefix)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	
	std::string fn = prefix + "timings.txt";
	std::experimental::filesystem::path p(fn);
	std::experimental::filesystem::create_directories(p.parent_path());
	std::ofstream ofout(fn.c_str());
	mclAssert(ofout.good(), "Logger::write_timings: failed to create file");
	ofout << print_timers();
	ofout.close();
}

void Logger::write_counters(std::string prefix)
{
    const std::lock_guard<std::mutex> lock(log_mutex);
	std::string fn = prefix + "counters.txt";
	std::experimental::filesystem::path p(fn);
	std::experimental::filesystem::create_directories(p.parent_path());
	std::ofstream ofout(fn.c_str());
	mclAssert(ofout.good(), "Logger::write_counters: failed to create file");
	ofout << print_counters();
	ofout.close();
}

template<class Archive> void Logger::serialize(Archive& archive)
{
	archive(
		curr_frame,
		print_timer_start,
		pause_simulation,
		per_frame_runtime_s,
		per_frame_data
	);
}

template void Logger::serialize<cereal::BinaryOutputArchive>(cereal::BinaryOutputArchive&);
template void Logger::serialize<cereal::BinaryInputArchive>(cereal::BinaryInputArchive&);

PerFrameData::PerFrameData()
{
}

int& PerFrameData::get_int_counter(const std::string& l)
{
	std::unordered_map<std::string, int>::iterator it = int_counters.find(l);
	if (it == int_counters.end()) { it = int_counters.emplace(l, 0).first; }
	return it->second;
}

double& PerFrameData::get_double_counter(const std::string& l)
{
	std::unordered_map<std::string, double>::iterator it = double_counters.find(l);
	if (it == double_counters.end()) { it = double_counters.emplace(l, 0).first; }
	return it->second;
}


double PerFrameData::get_time_ms(const std::string &l) const
{
	std::unordered_map<std::string,double>::const_iterator it = timings_ms.find(l);
	if (it == timings_ms.end())
		return 0;

	return it->second;
}

void PerFrameData::start_timer(const std::string &l)
{
	std::unordered_map<std::string,MicroTimer>::iterator it = timers.find(l);
	if (it == timers.end()) { it = timers.emplace(l, MicroTimer()).first; }
	it->second.reset(); // reset if the timer already exists
}

void PerFrameData::stop_timer(const std::string &l)
{
	std::unordered_map<std::string,MicroTimer>::iterator it = timers.find(l);
	if (it == timers.end()) // timer never started
		return;

	// Store elapsed time
	std::unordered_map<std::string,double>::iterator it2 = timings_ms.find(l);
	if (it2 == timings_ms.end()) { it2 = timings_ms.emplace(l, 0).first; }
	it2->second += it->second.elapsed_ms();
}

PerFrameData& Logger::get_frame_data(int iter)
{
	return per_frame_data.emplace(iter, PerFrameData()).first->second;
}

template<class Archive> void PerFrameData::serialize(Archive& archive)
{
	archive(
	 	int_counters,
	 	double_counters,
	 	timings_ms
	);
}

template void PerFrameData::serialize<cereal::BinaryOutputArchive>(cereal::BinaryOutputArchive&);
template void PerFrameData::serialize<cereal::BinaryInputArchive>(cereal::BinaryInputArchive&);

void SetCounterCurrentFrame::set_int(const char *l, int cnt)
{
	Logger &log = Logger::get();
	log.set_int_counter(log.curr_frame, std::string(l), cnt);
}

void SetCounterCurrentFrame::set_double(const char *l, double cnt)
{
	Logger &log = Logger::get();
	log.set_double_counter(log.curr_frame, std::string(l), cnt);
}

void SetCounterCurrentFrame::add_runtime(double cnt)
{
	Logger &log = Logger::get();
	log.add_runtime_s(log.curr_frame, cnt);
}

void SetCounterCurrentFrame::add_runtime(int iter, double cnt)
{
	Logger &log = Logger::get();
	log.add_runtime_s(iter, cnt);
}

FunctionTimerWrapper::FunctionTimerWrapper(const char *f_) : f(f_)
{
	Logger &log = Logger::get();
	log.start(log.curr_frame, f);
}
FunctionTimerWrapper::~FunctionTimerWrapper()
{
	Logger &log = Logger::get();
	log.stop(log.curr_frame, f);
}

LogFrameWrapper::LogFrameWrapper()
{
	Logger &log = Logger::get();
	log.curr_frame++;
	if (log.begin_frame != nullptr)
		log.begin_frame();
}

LogFrameWrapper::~LogFrameWrapper()
{
	Logger &log = Logger::get();
	log.add_runtime_s(log.curr_frame, timer.elapsed_s());

	if (log.end_frame != nullptr)
		log.end_frame();
}

} // ns mcl
