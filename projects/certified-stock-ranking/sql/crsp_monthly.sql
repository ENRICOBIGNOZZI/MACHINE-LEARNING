-- Template only. The Python adapter parameterizes table names and dates.
-- Common shares on NYSE/AMEX/Nasdaq, with delisting returns merged by month.
select a.permno, a.date, a.ret, a.retx, abs(a.prc) as prc, a.shrout, a.vol,
       n.shrcd, n.exchcd, d.dlret, d.dlstcd
from crsp.msf as a
left join crsp.msenames as n
  on a.permno=n.permno and n.namedt<=a.date and a.date<=n.nameendt
left join crsp.msedelist as d
  on a.permno=d.permno
 and date_trunc('month',a.date)=date_trunc('month',d.dlstdt)
where n.shrcd in (10,11) and n.exchcd in (1,2,3);
