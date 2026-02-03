#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <future>
#include <condition_variable>
#include <queue>
#include <vector>
#include <algorithm>
#include "FunctionTraits.hpp"
/**
 * @brief The ThreadQueue class
 */
class ThreadQueue
{
    using TimePoint = std::chrono::time_point<std::chrono::system_clock,std::chrono::nanoseconds>;

    friend class ThreadPool;
private:
    ThreadQueue()
    {
        m_Stop.store(false,std::memory_order_relaxed);
        m_Stoped.store(true,std::memory_order_relaxed);
        m_Thread = std::thread(&ThreadQueue::run,this);
    }

    ~ThreadQueue()
    {
        m_Stop.store(true,std::memory_order_relaxed);
        m_CV.notify_one();

        while (!m_Stoped.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }

        if (m_Thread.joinable())
            m_Thread.join();
    }

    ThreadQueue(const ThreadQueue&) = delete ;

    ThreadQueue(ThreadQueue&& other) = delete ;

    ThreadQueue& operator = (const ThreadQueue&) = delete ;

    ThreadQueue& operator = (ThreadQueue&&) = delete ;

    bool empty()
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        return m_TaskQue.empty();
    }

    std::size_t size()
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        return m_TaskQue.size();
    }

    void addTask(const std::function<void()>&& task)
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        bool isEmpty = this->m_TaskQue.empty();
        m_TaskQue.emplace(std::move(task));
        lock.unlock();

        //仅在队列为空的情况下才唤醒线程
        if(isEmpty)
            m_CV.notify_one();
    }

    //这个函数只能在ThreadQueue对象刚刚创建还没有开始执行任务的时候调用,否则目标对象other没有对任务队列加锁,是不安全的行为
    //所以这个函数仅仅只能用于转移任务队列到闲置的线程
    void moveTasks(ThreadQueue& other)
    {
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            other.m_TaskQue = std::move(this->m_TaskQue);
        }
        other.m_CV.notify_one();
    }

    bool occupied() const noexcept
    {
        bool occupyFlag = false;
        //如果起始时间大于结束时间,说明任务正在执行,如果执行时间大于10s,则进一步认为线程池被占用了,否则说明线程正在闲置
        if(m_Start > m_End)
        {
            std::chrono::nanoseconds interval = std::chrono::system_clock::now() - m_Start;
            double time = std::chrono::duration_cast<std::chrono::duration<double,std::ratio<1,1000>>>(interval).count();

            occupyFlag = time > 10*1000;
        }
        return occupyFlag;
    }

    bool isIdle()
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        return m_TaskQue.empty() && (m_End >= m_Start);
    }

    ///等待当前线程任务完成
    void wait(std::promise<bool>&& p)
    {
        std::unique_lock<std::mutex> lock(m_Mutex);
        if(m_TaskQue.empty() && (m_End >= m_Start))
        {
            p.set_value(true);
        }
        else
        {
            m_DoneFlag.store(true);
            m_Done = std::move(p);
        }
    }

    void run()
    {
        m_Stoped.store(false);
        while (!m_Stop.load(std::memory_order_relaxed))
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            if(m_TaskQue.empty())
            {
                if(m_DoneFlag.load())
                {
                    //m_DoneFlag控制线程只在调用了wait(std::promise<bool>&& p)之后设置一次promise
                    m_Done.set_value(true);
                    m_DoneFlag.store(false);
                }
                m_CV.wait(lock,[this](){return !m_TaskQue.empty() || m_Stop.load(std::memory_order_relaxed);});
            }
            else
            {
                std::function<void()> task = std::move(m_TaskQue.front());
                m_Start = std::chrono::system_clock::now();
                lock.unlock();

                task();

                std::unique_lock<std::mutex> lock(m_Mutex);
                m_TaskQue.pop();
                m_End = std::chrono::system_clock::now();
            }
        }
        m_Stoped.store(true);
    }

private:
    std::queue<std::function<void()>> m_TaskQue;
    std::thread m_Thread;
    std::mutex m_Mutex;
    std::condition_variable m_CV;
    std::atomic<bool> m_Stop;
    std::atomic<bool> m_Stoped;
    std::atomic<bool> m_DoneFlag{false};
    std::promise<bool> m_Done;
    TimePoint m_Start = std::chrono::system_clock::now();
    //结束时间小于或者等于起始时间都能代表线程闲置,由于系统缓存的原因,在重复执行同一个任务时有可能会在1ns内完成,此时两个时刻就是相等的
    TimePoint m_End =  std::chrono::system_clock::now();
};

/**
 * @brief The ThreadPool class
 *
 *如果在添加新的任务的时候有线程池被占用了,就创建一个新的线程池,然后将被占用线程池中的任务移动到新的线程池中
 *如果在添加新的任务时检测到之前被占用的线程池现在已经闲置了,就释放掉多余的线程池,使活跃的线程池数量尽量和CPU核心数保持一致
 */
class ThreadPool
{
public:
    enum Distribution{
        Ordered,//按线程池顺序依次分配任务
        Balanced//按线程池任务分布情况均匀地将任务分配给线程
    };

    ThreadPool(unsigned size = 0)
    {
        if(size >  std::thread::hardware_concurrency() || size == 0)
            size = std::thread::hardware_concurrency();

        for(unsigned i = 0; i < size; i++)
        {
            m_Threads.push_back(new ThreadQueue());
        }
        m_CurrentThread = m_Threads.begin();
    }

    ~ThreadPool()
    {
        std::vector<ThreadQueue*>::iterator it = m_Threads.begin();
        while (it != m_Threads.end())
        {
            //如果线程未被占用,就结束线程,**对于被占用的线程暂时不处理**
            if( !(*it)->occupied() )
            {
                delete (*it);
                ++it;
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete ;

    ThreadPool(ThreadPool&& other) = delete ;

    ThreadPool& operator = (const ThreadPool&) = delete ;

    ThreadPool& operator = (ThreadPool&&) = delete ;

    ///启动一个后台任务,返回值是一个与std::packaged_task相关联的future,当传入的函数抛出异常时异常会被保存到future中,因此不会对线程池的while循环造成破坏
    ///对future调用get()等同于同步执行任务,当前线程会阻塞直到后台任务完成并获取返回值
    ///不对future调用get()等同于异步执行任务,当前线程会继续向下执行并忽视返回值
    template<Distribution Mode = Ordered,typename Func,typename...Args,typename ReturnType = typename FunctionTraits<Func>::ReturnType>
    std::future<ReturnType> run(Func func,Args&&...args)
    {
        //1.检测是否存在新的被占用的线程
        detectNewIdleThread();

        //2.清除线程池中多余的闲置线程
        deleteIdleThread();

        //3.按要求查找一个未被占用的线程
        ThreadQueue* t = useableThread<Mode>();

        //4.封装任务并且将任务添加到线程队列中
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::bind(func,std::forward<Args>(args)...));
        std::future<ReturnType> future = task->get_future();
        t->addTask([task](){(*task)();});
        return future;
    }

    void waitforDone()
    {
        std::vector<ThreadQueue*>::iterator it = m_Threads.begin();
        while (it != m_Threads.end())
        {
            std::promise<bool> p;
            std::future<bool> f = p.get_future();
            (*it)->wait(std::move(p));
            f.get();

            ++it;
        }
    }

private:
    template<Distribution Mode>
    typename std::enable_if<Mode == Ordered,ThreadQueue*>::type
    useableThread()
    {
        //按顺序查找线程,返回一个未被占用的线程
        //按run函数中的顺序调用可以始终保证线程池中有未被占用的线程,因此不会陷入死循环
        while (true)
        {
            if(m_CurrentThread == m_Threads.end() )
                m_CurrentThread =  m_Threads.begin();

            if( (*m_CurrentThread)->occupied() )
                ++m_CurrentThread;
            else
                return *m_CurrentThread++;//返回当前线程指针并让这个指针指向下一个位置
        }
    }

    template<Distribution Mode>
    typename std::enable_if<Mode == Balanced,ThreadQueue*>::type
    useableThread()
    {
        //按任务数量查找线程,返回一个当前任务数量最少且未被占用的线程
        //按run函数中的顺序调用可以始终保证线程池中有未被占用的线程,因此不会返回错误指针
        auto it = std::min_element(m_Threads.cbegin(),m_Threads.cend(),[](ThreadQueue* t1,ThreadQueue* t2){
            const bool t1_avail = !t1->occupied();
            const bool t2_avail = !t2->occupied();

            if (t1_avail != t2_avail)
                return t1_avail;

            return t1_avail ? (t1->size() < t2->size()) : false;
        });
        return *it;
    }

    void deleteIdleThread()
    {
        if(m_Threads.size() > std::thread::hardware_concurrency())
        {
            std::vector<ThreadQueue*>::iterator it = m_Threads.begin();
            while (it != m_Threads.end())
            {
                if( (*it)->isIdle() )
                {
                    delete (*it);
                    it = m_Threads.erase(it);
                }
                else
                    ++it;

                if(m_Threads.size() <= std::thread::hardware_concurrency())
                    break;
            }
        }
    }

    void detectNewIdleThread()
    {
        //反向迭代,如果有被占用的线程可以将新建的线程添加到容器最后,避免遍历新添加的线程
        std::vector<ThreadQueue*>::reverse_iterator it = m_Threads.rbegin();
        while (it != m_Threads.rend())
        {
            if( (*it)->occupied() && !(*it)->empty() )
            {
                ThreadQueue* to = findIdleThread();
                if(to == nullptr)
                {
                    to = new ThreadQueue();
                    m_Threads.push_back(to);
                }

                (*it)->moveTasks(*to);
            }
            ++it;
        }
    }

    ThreadQueue* findIdleThread()
    {
        std::vector<ThreadQueue*>::iterator it = m_Threads.begin();
        while (it != m_Threads.end())
        {
            if( (*it)->isIdle() )
                return (*it);

            ++it;
        }

        return nullptr;
    }

private:
    std::vector<ThreadQueue*> m_Threads;
    std::vector<ThreadQueue*>::iterator m_CurrentThread;
};

#endif // THREADPOOL_H
