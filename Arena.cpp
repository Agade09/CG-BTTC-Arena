#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <array>
#include <random>
#include <chrono>
#include <omp.h>
#include <limits>
#include <algorithm>
#include <queue>
#include <thread>
#include <csignal>
using namespace std;
using namespace std::chrono;

constexpr bool Debug_AI{true},Timeout{false};
constexpr int PIPE_READ{0},PIPE_WRITE{1};
constexpr int N{2};//Number of players, 1v1
constexpr double FirstTurnTime{1*(Timeout?1:10)},TimeLimit{0.1*(Timeout?1:10)};
constexpr int H{20},W{35};

bool stop{false};//Global flag to stop all arena threads when SIGTERM is received

struct vec{
    int x,y;
    inline bool operator==(const vec &a)const noexcept{
        return x==a.x && y==a.y;
    }
    inline int idx()const noexcept{
        return y*W+x;
    }
    inline vec operator+(const vec &a)const noexcept{
        return vec{x+a.x,y+a.y};
    }
    inline vec operator/(const double a)const noexcept{
        return vec{static_cast<int>(round(x/a)),static_cast<int>(round(y/a))};
    }
    inline bool valid()const noexcept{
        return x>=0 && y>=0 && x<W && y<H;
    }
};

constexpr vec Neighbour_Vec[]{vec{1,0},vec{0,1},vec{-1,0},vec{0,-1},vec{1,-1},vec{1,1},vec{-1,1},vec{-1,-1}};

inline ostream& operator<<(ostream &os,const vec &r)noexcept{
    os << r.x << " " << r.y;
    return os;
}

inline istream& operator>>(istream &is,vec &r)noexcept{
    is >> r.x >> r.y;
    return is;
}

struct player{
    vec r;
    bool BIT;
    int id;
};

struct state{
    int turn;
    array<int,H*W> G;
    vector<player> P;
    vector<int> Score;
};

enum move_type{MOVE,BACK};

struct play{
    move_type type;
    vec target;
    int amount;
};

inline string EmptyPipe(const int fd){
    int nbytes;
    if(ioctl(fd,FIONREAD,&nbytes)<0){
        throw(4);
    }
    string out;
    out.resize(nbytes);
    if(read(fd,&out[0],nbytes)<0){
        throw(4);
    }
    return out;
}

struct AI{
    int id,pid,outPipe,errPipe,inPipe;
    string name;
    inline void stop(){
        if(alive()){
            kill(pid,SIGTERM);
            int status;
            waitpid(pid,&status,0);//It is necessary to read the exit code for the process to stop
            if(!WIFEXITED(status)){//If not exited normally try to "kill -9" the process
                kill(pid,SIGKILL);
            }
        }
    }
    inline bool alive()const{
        return kill(pid,0)!=-1;//Check if process is still running
    }
    inline void Feed_Inputs(const string &inputs){
        if(write(inPipe,&inputs[0],inputs.size())!=inputs.size()){
            throw(5);
        }
    }
    inline ~AI(){
        close(errPipe);
        close(outPipe);
        close(inPipe);
        stop();
    }
};

void StartProcess(AI &Bot){
    int StdinPipe[2];
    int StdoutPipe[2];
    int StderrPipe[2];
    if(pipe(StdinPipe)<0){
        perror("allocating pipe for child input redirect");
    }
    if(pipe(StdoutPipe)<0){
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        perror("allocating pipe for child output redirect");
    }
    if(pipe(StderrPipe)<0){
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        perror("allocating pipe for child stderr redirect failed");
    }
    int nchild{fork()};
    if(nchild==0){//Child process
        if(dup2(StdinPipe[PIPE_READ],STDIN_FILENO)==-1){// redirect stdin
            perror("redirecting stdin");
            return;
        }
        if(dup2(StdoutPipe[PIPE_WRITE],STDOUT_FILENO)==-1){// redirect stdout
            perror("redirecting stdout");
            return;
        }
        if(dup2(StderrPipe[PIPE_WRITE],STDERR_FILENO)==-1){// redirect stderr
            perror("redirecting stderr");
            return;
        }
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        close(StderrPipe[PIPE_READ]);
        close(StderrPipe[PIPE_WRITE]);
        execl(Bot.name.c_str(),Bot.name.c_str(),(char*)NULL);//(char*)Null is really important
        //If you get past the previous line its an error
        perror("exec of the child process");
    }
    else if(nchild>0){//Parent process
        close(StdinPipe[PIPE_READ]);//Parent does not read from stdin of child
        close(StdoutPipe[PIPE_WRITE]);//Parent does not write to stdout of child
        close(StderrPipe[PIPE_WRITE]);//Parent does not write to stderr of child
        Bot.inPipe=StdinPipe[PIPE_WRITE];
        Bot.outPipe=StdoutPipe[PIPE_READ];
        Bot.errPipe=StderrPipe[PIPE_READ];
        Bot.pid=nchild;
    }
    else{//failed to create child
        close(StdinPipe[PIPE_READ]);
        close(StdinPipe[PIPE_WRITE]);
        close(StdoutPipe[PIPE_READ]);
        close(StdoutPipe[PIPE_WRITE]);
        perror("Failed to create child process");
    }
}

inline bool IsValidMove(const state &S,const AI &Bot,const string &M){
    return count(M.begin(),M.end(),'\n')==1;
}

string GetMove(const state &S,AI &Bot,const int turn){
    pollfd outpoll{Bot.outPipe,POLLIN};
    time_point<system_clock> Start_Time{system_clock::now()};
    string out;
    while(static_cast<duration<double>>(system_clock::now()-Start_Time).count()<(turn==1?FirstTurnTime:TimeLimit) && !IsValidMove(S,Bot,out)){
        double TimeLeft{(turn==1?FirstTurnTime:TimeLimit)-static_cast<duration<double>>(system_clock::now()-Start_Time).count()};
        if(poll(&outpoll,1,TimeLeft)){
            out+=EmptyPipe(Bot.outPipe);
        }
    }
    return out;
}

inline bool Has_Won(const array<AI,N> &Bot,const int idx)noexcept{
    if(!Bot[idx].alive()){
        return false;
    }
    for(int i=0;i<N;++i){
        if(i!=idx && Bot[i].alive()){
            return false;
        }
    }
    return true;
}

inline bool All_Dead(const array<AI,N> &Bot)noexcept{
    for(const AI &b:Bot){
        if(b.alive()){
            return false;
        }
    }
    return true;
}

void Simulate(state &S,const array<play,N> &M){
    for(int i=0;i<N;++i){
        const play& mv=M[i];
        if(mv.type==MOVE){
            if(mv.target.x!=S.P[i].r.x){
                S.P[i].r.x+=S.P[i].r.x<mv.target.x?1:-1;
            }
            else if(mv.target.y!=S.P[i].r.y){
                S.P[i].r.y+=S.P[i].r.y<mv.target.y?1:-1;
            }
        }
        else if(mv.type==BACK){
            S.P[i].BIT=false;
        }
    }
    for(int i=0;i<N;++i){
        if(S.G[S.P[i].r.idx()]==-1 && find_if(S.P.begin(),S.P.end(),[&](const player &p){return p.r==S.P[i].r && p.id!=S.P[i].id;})==S.P.end()){
            S.G[S.P[i].r.idx()]=S.P[i].id;
            ++S.Score[i];
        }
    }
    array<bool,H*W> visited;
    fill(visited.begin(),visited.end(),false);
    for(int i=0;i<H*W;++i){
        if(!visited[i] && S.G[i]==-1){
            int color{-1};
            const vec start{i%W,i/W};
            queue<vec> bfs_queue;
            bfs_queue.push(start);
            vector<int> Neutrals;
            Neutrals.push_back(i);
            bool encircle{true};
            visited[i]=true;
            while(!bfs_queue.empty()){
                vec r=bfs_queue.front();
                bfs_queue.pop();
                for(const vec &dr:Neighbour_Vec){
                    vec candidate=r+dr;
                    if(candidate.valid()){
                        if(S.G[candidate.idx()]==-1){
                            if(!visited[candidate.idx()]){
                                bfs_queue.push(candidate);
                                Neutrals.push_back(candidate.idx());
                                visited[candidate.idx()]=true;  
                            }
                        }
                        else{
                            if(color==-1){
                                color=S.G[candidate.idx()];
                            }
                            else if(S.G[candidate.idx()]!=color){
                                encircle=false; 
                            }
                        } 
                    }
                    else{
                        encircle=false;
                    }
                }
            }
            if(encircle){
                for(const int a:Neutrals){
                    S.G[a]=color;
                }
                S.Score[color]+=Neutrals.size();
            }
        }
    }
    ++S.turn;
}

play StringToStrat(const state &S,const AI &Bot,const string &M_str){
    stringstream ss(M_str);
    string type;
    ss >> type;
    play mv;
    if(type=="BACK"){
        mv.type=BACK;
        ss >> mv.amount;
        if(mv.amount>25 || !S.P[Bot.id].BIT){
            throw(1);
        }
    }
    else if(find_if(type.begin(),type.end(),[&](const char c){return !isdigit(c);})==type.end() && find_if(type.begin(),type.end(),[&](const char c){return isdigit(c);})!=type.end()){
        stringstream ss2(M_str);
        mv.type=MOVE;
        ss2 >> mv.target;
        if(!mv.target.valid()){
            throw(1);
        }
    }
    else{
        throw(1);
    }
    return mv;
}

bool Winner_Cant_Change(const state &S){
    return S.Score[0]>H*W/2 || S.Score[1]>H*W/2;
}

int Play_Game(const array<string,N> &Bot_Names,state &S){
    array<AI,N> Bot;
    for(int i=0;i<N;++i){
        Bot[i].id=i;
        Bot[i].name=Bot_Names[i];
        StartProcess(Bot[i]);
        stringstream ss;
        ss << N-1 << endl;
        Bot[i].Feed_Inputs(ss.str());
    }
    while(!stop){
        array<play,N> M;
        for(int i=0;i<N;++i){
            if(Bot[i].alive()){
                stringstream ss;
                ss << S.turn << endl;
                ss << S.P[i].r << " " << S.P[i].BIT << endl;
                for(int j=0;j<N;++j){
                    if(j!=i){
                        ss << S.P[j].r << " " << S.P[j].BIT << endl;
                    }
                }
                for(int y=0;y<H;++y){
                    for(int x=0;x<W;++x){
                        const int& a{S.G[y*W+x]};
                        if(a==-1){
                            ss << '.';
                        }
                        else{
                            ss << (a==i?0:a>i?a-1:a+1);
                        }
                    }
                    ss << endl;
                }
                try{
                    Bot[i].Feed_Inputs(ss.str());
                    M[i]=StringToStrat(S,Bot[i],GetMove(S,Bot[i],S.turn));
                    //cerr << M[i] << endl;
                }
                catch(int ex){
                    if(ex==1){//Timeout
                        cerr << "Loss by Timeout of AI " << Bot[i].id << " name: " << Bot[i].name << endl;
                    }
                    else if(ex==5){
                        cerr << "AI " << Bot[i].name << " died before being able to give it inputs" << endl;
                    }
                    Bot[i].stop();
                }
            }
        }
        for(int i=0;i<N;++i){
            string err_str{EmptyPipe(Bot[i].errPipe)};
            if(Debug_AI){
                ofstream err_out("log.txt",ios::app);
                err_out << err_str << endl;
            }
            if(Has_Won(Bot,i)){
                //cerr << i << " has won in " << turn << " turns" << endl;
                return i;
            }
        }
        if(All_Dead(Bot)){
            return -1;
        }
        Simulate(S,M);
        if(S.turn==350 || Winner_Cant_Change(S)){
            return S.Score[0]>S.Score[1]?0:S.Score[1]>S.Score[0]?1:-1;
        }
    }
    return -2;
}

int Play_Round(array<string,N> Bot_Names){
    default_random_engine generator(system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> Swap_Distrib(0,1);
    const bool player_swap{Swap_Distrib(generator)==1};
    if(player_swap){
        swap(Bot_Names[0],Bot_Names[1]);
    }
    state S;
    S.P.resize(N);
    S.Score.resize(N);
    fill(S.G.begin(),S.G.end(),-1);
    fill(S.Score.begin(),S.Score.end(),1);
    uniform_int_distribution<int> X_Distrib(0,W-1),Y_Distrib(0,H-1);
    for(int i=0;i<N;++i){
        do{
            S.P[i].r=vec{X_Distrib(generator),Y_Distrib(generator)};  
        }while(find_if(S.P.begin(),next(S.P.begin(),i),[&](const player &p){return p.r==S.P[i].r;})!=next(S.P.begin(),i));
        S.P[i].BIT=true;
        S.P[i].id=i;
        S.G[S.P[i].r.idx()]=i;
    }
    S.turn=1;
    int winner{Play_Game(Bot_Names,S)};
    if(player_swap){
        return winner==-1?-1:winner==0?1:0;
    }
    else{
        return winner;
    }
}

void StopArena(const int signum){
    stop=true;
}

int main(int argc,char **argv){
    if(argc<3){
        cerr << "Program takes 2 inputs, the names of the AIs fighting each other" << endl;
        return 0;
    }
    int N_Threads{1};
    if(argc>=4){//Optional N_Threads parameter
        N_Threads=min(2*omp_get_num_procs(),max(1,atoi(argv[3])));
        cerr << "Running " << N_Threads << " arena threads" << endl;
    }
    array<string,N> Bot_Names;
    for(int i=0;i<2;++i){
        Bot_Names[i]=argv[i+1];
    }
    cout << "Testing AI " << Bot_Names[0];
    for(int i=1;i<N;++i){
        cerr << " vs " << Bot_Names[i];
    }
    cerr << endl;
    for(int i=0;i<N;++i){//Check that AI binaries are present
        ifstream Test{Bot_Names[i].c_str()};
        if(!Test){
            cerr << Bot_Names[i] << " couldn't be found" << endl;
            return 0;
        }
        Test.close();
    }
    signal(SIGTERM,StopArena);//Register SIGTERM signal handler so the arena can cleanup when you kill it
    signal(SIGPIPE,SIG_IGN);//Ignore SIGPIPE to avoid the arena crashing when an AI crashes
    int games{0},draws{0};
    array<double,2> points{0,0};
    #pragma omp parallel num_threads(N_Threads) shared(games,points,Bot_Names)
    while(!stop){
        int winner{Play_Round(Bot_Names)};
        if(winner==-1){//Draw
            #pragma omp atomic
            ++draws;
            #pragma omp atomic
            points[0]+=0.5;
            #pragma omp atomic
            points[1]+=0.5;
        }
        else{//Win
            ++points[winner];
        }
        #pragma omp atomic
        ++games;
        double p{static_cast<double>(points[0])/games};
        double sigma{sqrt(p*(1-p)/games)};
        double better{0.5+0.5*erf((p-0.5)/(sqrt(2)*sigma))};
        #pragma omp critical
        cout << "Wins:" << setprecision(4) << 100*p << "+-" << 100*sigma << "% Rounds:" << games << " Draws:" << draws << " " << better*100 << "% chance that " << Bot_Names[0] << " is better" << endl;
    }
}