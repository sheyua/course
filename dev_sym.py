#!/usr/bin/python3
import sys; sys.path.append('./utils')
import sym

sym.restart_syms_min(sym.all_ex_lst())
sym.restart_syms_day(sym.liq_lst())
sym.resume_syms_day(sym.liq_lst())
sym.restart_trends_day(sym.trend_lst())
sym.resume_trends_day(sym.trend_lst())

#sym.resume_syms_min(sym.liq_lst())
#sym.resume_syms_day(sym.liq_lst())
#sym.resume_trends_day(sym.trend_lst())
