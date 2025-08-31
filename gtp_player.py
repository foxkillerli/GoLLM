#!/usr/bin/env python3
from __future__ import annotations
import argparse
import socketserver
from typing import List, Optional, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

_GTP_COLS = [*"ABCDEFGHJKLMNOPQRSTUVWXYZ"]

# ---------------- Coordinate helpers ----------------

def gtp_str_to_xy(s: str, size: int) -> Optional[Tuple[int, int]]:
    s = s.strip().upper()
    if s in ("PASS", "PA", "P"):
        return None
    col_ch = s[0]
    row = int(s[1:])
    x = _GTP_COLS.index(col_ch) + 1
    return (x, row)

def xy_to_gtp_str(x: int, y: int, size: int) -> str:
    return f"{_GTP_COLS[x-1]}{y}"

def xy_to_sgf_token(x: int, y: int, size: int) -> str:
    y_top = size - y + 1
    return chr(ord('a')+x-1) + chr(ord('a')+y_top-1)

def sgf_token_to_xy(tok: str, size: int) -> Optional[Tuple[int,int]]:
    if tok == 'ZP':
        return None
    x = ord(tok[0])-ord('a')+1
    y_top = ord(tok[1])-ord('a')+1
    y = size-y_top+1
    return (x,y)

# ---------------- Board ----------------

class GoBoard:
    def __init__(self, size: int=19):
        self.size=size; self.clear()
    def clear(self):
        self.grid=[[None for _ in range(self.size)] for _ in range(self.size)]
        self.history=[]; self.to_play='B'; self.komi=7.5
    def neighbors(self,x,y):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 1<=nx<=self.size and 1<=ny<=self.size:
                yield nx,ny
    def get_group(self,x,y):
        color=self.grid[y-1][x-1]
        if color is None: return set(),set()
        stones={(x,y)};libs=set();stack=[(x,y)]
        while stack:
            cx,cy=stack.pop()
            for nx,ny in self.neighbors(cx,cy):
                v=self.grid[ny-1][nx-1]
                if v is None: libs.add((nx,ny))
                elif v==color and (nx,ny) not in stones:
                    stones.add((nx,ny)); stack.append((nx,ny))
        return stones,libs
    def legal_moves(self,color):
        moves=[]
        for y in range(1,self.size+1):
            for x in range(1,self.size+1):
                if self.grid[y-1][x-1] is None:
                    moves.append((x,y))
        moves.append(None); return moves
    def play(self,color,move):
        color=color.upper(); opp='W' if color=='B' else 'B'
        if move is not None:
            x,y=move
            if self.grid[y-1][x-1] is not None:
                raise ValueError("occupied")
            self.grid[y-1][x-1]=color
            captured_any=False;to_remove=set()
            for nx,ny in self.neighbors(x,y):
                if self.grid[ny-1][nx-1]==opp:
                    stones,libs=self.get_group(nx,ny)
                    if not libs: to_remove|=stones;captured_any=True
            for rx,ry in to_remove: self.grid[ry-1][rx-1]=None
            stones,libs=self.get_group(x,y)
            if not libs and not captured_any:
                self.grid[y-1][x-1]=None; raise ValueError("suicide")
        self.history.append((color,move)); self.to_play='W' if color=='B' else 'B'
    def moves_as_tokens(self):
        return ['ZP' if mv is None else xy_to_sgf_token(mv[0],mv[1],self.size) for _,mv in self.history]

# ---------------- Policy ----------------

class LLMPolicy:
    def __init__(self,base_model,adapter=None,device='auto',dtype=None,candidate_batch=64):
        self.base_model=base_model; self.adapter=adapter; self.device_arg=device; self.dtype=dtype
        self.candidate_batch=candidate_batch; self.tokenizer=None; self.model=None; self.device=None
    def load(self):
        self.tokenizer=AutoTokenizer.from_pretrained(self.base_model,use_fast=True,trust_remote_code=True)
        need_resize=False
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token: self.tokenizer.pad_token=self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token':'[PAD]'}); need_resize=True
        self.tokenizer.padding_side='left'
        torch_dtype=None
        if self.dtype: torch_dtype=getattr(torch,self.dtype)
        self.device=torch.device('cuda' if (self.device_arg=='auto' and torch.cuda.is_available()) else self.device_arg)
        self.model=AutoModelForCausalLM.from_pretrained(self.base_model,torch_dtype=torch_dtype,low_cpu_mem_usage=True,trust_remote_code=True).to(self.device)
        if need_resize: self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id=self.tokenizer.pad_token_id
        if self.adapter:
            if not _HAS_PEFT: raise RuntimeError("peft not installed")
            self.model=PeftModel.from_pretrained(self.model,self.adapter)
        self.model.eval()
    @torch.no_grad()
    def score_candidates(self, ctx: str, candidates: List[str]) -> Dict[str, float]:
        if not candidates: return {}
        ctx_ids=self.tokenizer(ctx, add_special_tokens=False).input_ids
        ctx_len=len(ctx_ids)
        scores: Dict[str,float]={}
        bs=self.candidate_batch
        for i in range(0,len(candidates),bs):
            batch=candidates[i:i+bs]
            texts=[ctx+t for t in batch]
            enc=self.tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False)
            input_ids=enc['input_ids'].to(self.device); attn=enc['attention_mask'].to(self.device)
            out=self.model(input_ids,attention_mask=attn)
            logits=out.logits; logprobs=torch.log_softmax(logits[:,:-1,:],dim=-1)
            labels=input_ids[:,1:]
            for b, tok in enumerate(batch):
                T=int(attn[b].sum().item())
                suff_len=max(0, int(T) - 1 - (ctx_len-0))
                if suff_len<=0:
                    scores[tok]=-1e30; continue
                start=(T-1)-suff_len; end=T-1
                lp=logprobs[b,start:end,:]; lab=labels[b,start:end]
                gathered=lp.gather(-1,lab.unsqueeze(-1)).squeeze(-1)
                scores[tok]=gathered.mean().item()
        return scores
    @torch.no_grad()
    def choose_move_token(self,ctx,candidates):
        scores=self.score_candidates(ctx,candidates)
        if not scores:
            return candidates[0] if candidates else 'ZP'
        return max(scores.items(), key=lambda kv: kv[1])[0]
    @torch.no_grad()
    def choose_move_token(self,ctx,candidates):
        ctx_ids=self.tokenizer(ctx,return_tensors='pt',add_special_tokens=False)['input_ids'].to(self.device)
        best_tok, best_score=candidates[0], -1e30
        bs=self.candidate_batch
        for i in range(0,len(candidates),bs):
            batch=candidates[i:i+bs]
            texts=[ctx+t for t in batch]
            enc=self.tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False)
            input_ids=enc['input_ids'].to(self.device); attn=enc['attention_mask'].to(self.device)
            out=self.model(input_ids,attention_mask=attn)
            logits=out.logits; logprobs=torch.log_softmax(logits[:,:-1,:],dim=-1)
            labels=input_ids[:,1:]
            indiv=[self.tokenizer(ctx+t,add_special_tokens=False).input_ids for t in batch]
            ctx_len=len(self.tokenizer(ctx,add_special_tokens=False).input_ids)
            suffix_lens=[len(x)-ctx_len for x in indiv]
            scores=[]
            for b,suff_len in enumerate(suffix_lens):
                T=int(attn[b].sum()); start=(T-1)-suff_len; end=T-1
                lp=logprobs[b,start:end,:]; lab=labels[b,start:end]
                gathered=lp.gather(-1,lab.unsqueeze(-1)).squeeze(-1)
                score=gathered.mean().item() if suff_len>0 else -1e30
                scores.append(score)
            for tok,sc in zip(batch,scores):
                if sc>best_score: best_score, best_tok=sc,tok
        return best_tok

# ---------------- Engine ----------------

OPENING_POINTS=[(4,4),(16,4),(4,16),(16,16),(3,4),(4,3),(16,3),(3,16)]

class Engine:
    def __init__(self,policy:LLMPolicy,alpha:float=0.7):
        self.policy=policy; self.alpha=alpha
    def tactical_candidates(self,board:GoBoard,color:str)->List[str]:
        opp='W' if color=='B' else 'B'; cand=set(); size=board.size
        for y in range(1,size+1):
            for x in range(1,size+1):
                if board.grid[y-1][x-1]==opp:
                    stones,libs=board.get_group(x,y)
                    if len(libs)==1: cand.add(next(iter(libs)))
        for y in range(1,size+1):
            for x in range(1,size+1):
                if board.grid[y-1][x-1]==color:
                    stones,libs=board.get_group(x,y)
                    if len(libs)==1: cand.add(next(iter(libs)))
        if board.history and board.history[-1][1]:
            lx,ly=board.history[-1][1]
            for yy in range(max(1,ly-2),min(size,ly+2)+1):
                for xx in range(max(1,lx-2),min(size,lx+2)+1):
                    if board.grid[yy-1][xx-1] is None: cand.add((xx,yy))
        return [xy_to_sgf_token(x,y,size) for (x,y) in cand if board.grid[y-1][x-1] is None]+['ZP']
    def opening_candidates(self,board:GoBoard)->List[str]:
        if len(board.history)<6:
            toks=[xy_to_sgf_token(x,y,board.size) for x,y in OPENING_POINTS if board.grid[y-1][x-1] is None]
            return toks
        return []
    def reading_score(self,board:GoBoard,move:Tuple[int,int],color:str)->float:
        try:
            tmp=GoBoard(board.size)
            tmp.grid=[row[:] for row in board.grid]; tmp.history=board.history[:]
            tmp.play(color,move)
            opp='W' if color=='B' else 'B'; score=0.0
            for nx,ny in tmp.neighbors(*move):
                if tmp.grid[ny-1][nx-1]==opp:
                    _,libs=tmp.get_group(nx,ny)
                    if not libs: score+=1.0
            return score
        except: return 0.0
    def propose(self,board:GoBoard,color:str):
        ctx=''.join(board.moves_as_tokens())
        cand=set(self.tactical_candidates(board,color))|set(self.opening_candidates(board))
        if len(cand)<5:
            for y in range(1,board.size+1):
                for x in range(1,board.size+1):
                    if board.grid[y-1][x-1] is None: cand.add(xy_to_sgf_token(x,y,board.size))
            cand.add('ZP')
        candidates=sorted(cand)
        # tactic scores/ reading_score
        t_scores: Dict[str,float]={}
        for tok in candidates:
            if tok=='ZP': t_scores[tok]=0.0
            else:
                xy=sgf_token_to_xy(tok,board.size)
                t_scores[tok]=self.reading_score(board,xy,color)
        # llm_scores
        llm_scores=self.policy.score_candidates(ctx,candidates)
        # normalisation
        def norm(d:Dict[str,float]):
            if not d: return {k:0.0 for k in candidates}
            v=list(d.values()); mn=min(v); mx=max(v)
            if mx-mn<1e-6: return {k:0.0 for k in d}
            return {k:(d[k]-mn)/(mx-mn) for k in d}
        tn=norm(t_scores); ln=norm(llm_scores)
        fused={k: self.alpha*tn.get(k,0.0) + (1.0-self.alpha)*ln.get(k,0.0) for k in candidates}
        # Avoid premature PASS
        for k in fused:
            if k == 'ZP':
                fused[k] -= 0.5
        best_tok=max(fused.items(), key=lambda kv: kv[1])[0]
        move=sgf_token_to_xy(best_tok,board.size)
        return move,best_tok

# ---------------- GTP Handler ----------------

class GTPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        engine=self.server.engine; board=GoBoard(size=self.server.init_board_size)
        while True:
            line=self.rfile.readline()
            if not line: break
            line=line.decode().strip()
            if not line: continue
            try: resp=self.dispatch(line,board,engine)
            except Exception as e: resp=f"? {e}"
            self.wfile.write((resp+"\n\n").encode()); self.wfile.flush()
    def dispatch(self,line,board,engine):
        parts=line.split();
        if not parts: return "? empty"
        cmd,args=parts[0].lower(),parts[1:]
        if cmd=='protocol_version': return "= 2"
        if cmd=='name': return f"= LLMGo"
        if cmd=='version': return f"= 0.2"
        if cmd=='boardsize': board.clear(); board.size=int(args[0]); return "="
        if cmd=='clear_board': board.clear(); return "="
        if cmd=='play':
            c=args[0].upper(); mv=gtp_str_to_xy(args[1],board.size); board.play(c,mv); return "="
        if cmd=='genmove':
            c=args[0].upper(); mv,tok=engine.propose(board,c)
            if mv is None: gtp='PASS'
            else: gtp=xy_to_gtp_str(mv[0],mv[1],board.size)
            board.play(c,mv); return f"= {gtp}"
        if cmd=='showboard':
            out=[]; cols=' '.join(_GTP_COLS[:board.size]); out.append('   '+cols)
            for y in range(board.size,0,-1):
                row=[ '.' if board.grid[y-1][x-1] is None else ('X' if board.grid[y-1][x-1]=='B' else 'O') for x in range(1,board.size+1)]
                out.append(f"{y:2d} "+' '.join(row))
            return "=\n"+'\n'.join(out)
        if cmd=='quit': return "="
        return f"? unknown {cmd}"

class ThreadedTCPServer(socketserver.ThreadingMixIn,socketserver.TCPServer): daemon_threads=True; allow_reuse_address=True
class GTPServer(ThreadedTCPServer):
    def __init__(self,addr,handler,engine,init_board_size): super().__init__(addr,handler); self.engine=engine; self.init_board_size=init_board_size

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--base-model',required=True); p.add_argument('--adapter',default=None); p.add_argument('--device',default='auto');
    p.add_argument('--dtype',default=None); p.add_argument('--candidate-batch',type=int,default=64);
    p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=5001); p.add_argument('--board-size',type=int,default=19);
    p.add_argument('--alpha',type=float,default=0.7)
    args=p.parse_args()
    pol=LLMPolicy(args.base_model,args.adapter,args.device,args.dtype,args.candidate_batch); pol.load()
    eng=Engine(pol,alpha=args.alpha)
    with GTPServer((args.host,args.port),GTPHandler,eng,args.board_size) as srv:
        print(f"[GTP] Listening on {args.host}:{args.port}"); srv.serve_forever()

if __name__=='__main__': main()
