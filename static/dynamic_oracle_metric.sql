SET markup html ON spool ON pre off entmap off

set term off
set heading on
set verify off
set feedback off

set linesize 2000
set pagesize 30000
set long 999999999
set longchunksize 999999

column index_name format a30
column table_name format a30
column num_rows format 999999999
column index_type format a24
column num_rows format 999999999
column status format a8
column clustering_factor format 999999999
column degree format a10
column blevel format 9
column distinct_keys format 9999999999
column leaf_blocks format   9999999
column last_analyzed    format a10
column column_name format a25
column column_position format 9
column temporary format a2
column partitioned format a5
column partitioning_type format a7
column partition_count format 999
column program  format a30
column spid  format a6
column pid  format 99999
column sid  format 99999
column serial# format 99999
column username  format a12
column osuser    format a12
column logon_time format  date
column event    format a32
column JOB_NAME        format a30
column PROGRAM_NAME    format a32
column STATE           format a10
column window_name           format a30
column repeat_interval       format a60
column machine format a30
column program format a30
column osuser format a15
column username format a15
column event format a50
column seconds format a10
column sqltext format a100



column dbid new_value spool_dbid
column inst_num new_value spool_inst_num
select dbid from v$database where rownum = 1;
select instance_number as inst_num from v$instance where rownum = 1;
column spoolfile_name new_value spoolfile
select 'spool_'||(select name from v$database where rownum=1) ||'_'|| (select instance_name from v$instance where rownum=1)||'_'||to_char(sysdate,'yy-mm-dd_hh24.mi')||'_dynamic' as spoolfile_name from dual;
spool &&spoolfile..html

prompt <p>awr视图中的load profile
select s.snap_date,
       decode(s.redosize, null, '--shutdown or end--', s.currtime) "TIME",
       to_char(round(s.seconds/60,2)) "elapse(min)",
       round(t.db_time / 1000000 / 60, 2) "DB time(min)",
       s.redosize redo,
       round(s.redosize / s.seconds, 2) "redo/s",
       s.logicalreads logical,
       round(s.logicalreads / s.seconds, 2) "logical/s",
       physicalreads physical,
       round(s.physicalreads / s.seconds, 2) "phy/s",
       s.executes execs,
       round(s.executes / s.seconds, 2) "execs/s",
       s.parse,
       round(s.parse / s.seconds, 2) "parse/s",
       s.hardparse,
       round(s.hardparse / s.seconds, 2) "hardparse/s",
       s.transactions trans,
       round(s.transactions / s.seconds, 2) "trans/s"
  from (select curr_redo - last_redo redosize,
               curr_logicalreads - last_logicalreads logicalreads,
               curr_physicalreads - last_physicalreads physicalreads,
               curr_executes - last_executes executes,
               curr_parse - last_parse parse,
               curr_hardparse - last_hardparse hardparse,
               curr_transactions - last_transactions transactions,
               round(((currtime + 0) - (lasttime + 0)) * 3600 * 24, 0) seconds,
               to_char(currtime, 'yy/mm/dd') snap_date,
               to_char(currtime, 'hh24:mi') currtime,
               currsnap_id endsnap_id,
               to_char(startup_time, 'yyyy-mm-dd hh24:mi:ss') startup_time
          from (select a.redo last_redo,
                       a.logicalreads last_logicalreads,
                       a.physicalreads last_physicalreads,
                       a.executes last_executes,
                       a.parse last_parse,
                       a.hardparse last_hardparse,
                       a.transactions last_transactions,
                       lead(a.redo, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_redo,
                       lead(a.logicalreads, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_logicalreads,
                       lead(a.physicalreads, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_physicalreads,
                       lead(a.executes, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_executes,
                       lead(a.parse, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_parse,
                       lead(a.hardparse, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_hardparse,
                       lead(a.transactions, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_transactions,
                       b.end_interval_time lasttime,
                       lead(b.end_interval_time, 1, null) over(partition by b.startup_time order by b.end_interval_time) currtime,
                       lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) currsnap_id,
                       b.startup_time
                  from (select snap_id,
                               dbid,
                               instance_number,
                               sum(decode(stat_name, 'redo size', value, 0)) redo,
                               sum(decode(stat_name,
                                          'session logical reads',
                                          value,
                                          0)) logicalreads,
                               sum(decode(stat_name,
                                          'physical reads',
                                          value,
                                          0)) physicalreads,
                               sum(decode(stat_name, 'execute count', value, 0)) executes,
                               sum(decode(stat_name,
                                          'parse count (total)',
                                          value,
                                          0)) parse,
                               sum(decode(stat_name,
                                          'parse count (hard)',
                                          value,
                                          0)) hardparse,
                               sum(decode(stat_name,
                                          'user rollbacks',
                                          value,
                                          'user commits',
                                          value,
                                          0)) transactions
                          from dba_hist_sysstat
                         where stat_name in
                               ('redo size',
                                'session logical reads',
                                'physical reads',
                                'execute count',
                                'user rollbacks',
                                'user commits',
                                'parse count (hard)',
                                'parse count (total)')
                         group by snap_id, dbid, instance_number) a,
                       dba_hist_snapshot b
                 where a.snap_id = b.snap_id
                   and a.dbid = b.dbid
                   and a.instance_number = b.instance_number
                   and a.dbid = &&spool_dbid
                   and a.instance_number = &&spool_inst_num
                 order by end_interval_time)) s,
       (select lead(a.value, 1, null) over(partition by b.startup_time order by b.end_interval_time) - a.value db_time,
               lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) endsnap_id
          from dba_hist_sys_time_model a, dba_hist_snapshot b
         where a.snap_id = b.snap_id
           and a.dbid = b.dbid
           and a.instance_number = b.instance_number
           and a.stat_name = 'DB time'
           and a.dbid = &&spool_dbid
           and a.instance_number = &&spool_inst_num) t
 where s.endsnap_id = t.endsnap_id
 order by  s.snap_date desc ,time asc;

prompt <p>IOPS 和 吞吐量的 每秒值
select "Time+Delta", "Metric", 
       case when "Total" >10000000 then '* '||round("Total"/1024/1024,0)||' M' 
            when "Total" between 10000 and 10000000 then '+ '||round("Total"/1024,0)||' K'
            when "Total" between 10 and 1024 then '  '||to_char(round("Total",0))
            else '  '||to_char("Total") 
       end "Total"
from (
 select to_char(min(begin_time),'hh24:mi:ss')||' /'||round(avg(intsize_csec/100),0)||'s' "Time+Delta",
       metric_name||' - '||metric_unit "Metric", 
       nvl(sum(value_inst1),0)+nvl(sum(value_inst2),0) "Total",
       sum(value_inst1) inst1, sum(value_inst2) inst2
 from
  ( select begin_time,intsize_csec,metric_name,metric_unit,metric_id,group_id,
       case inst_id when 1 then round(value,1) end value_inst1,
       case inst_id when 2 then round(value,1) end value_inst2
  from gv$sysmetric
  where metric_name in ('Host CPU Utilization (%)','Current OS Load', 'Physical Write Total IO Requests Per Sec', 
        'Physical Write Total Bytes Per Sec', 'Physical Write IO Requests Per Sec', 'Physical Write Bytes Per Sec',
         'I/O Requests per Second', 'I/O Megabytes per Second',
        'Physical Read Total Bytes Per Sec', 'Physical Read Total IO Requests Per Sec', 'Physical Read IO Requests Per Sec',
        'CPU Usage Per Sec','Network Traffic Volume Per Sec','Logons Per Sec','Redo Generated Per Sec','Redo Writes Per Sec',
        'User Transaction Per Sec','Average Active Sessions','Average Synchronous Single-Block Read Latency',
        'Logical Reads Per Sec','DB Block Changes Per Sec')
  )
 group by metric_id,group_id,metric_name,metric_unit
 order by metric_name
);

prompt <p>按照SNAP SHOT的时间统计IOPS和吞吐量、
select  
       s.instance_number,
       s.snap_date,
       to_char(round(s.seconds/60,2)) "elapse(min)",
       round(t.db_time / 1000000 / 60, 2) "DB time(min)",
       (s.phy_total_read_req +s.phy_read_mul_req+s.phy_total_write_req+s.phy_write_mul_req+s.redo_writes) IOPS,
       round((s.phy_total_read_req+s.phy_read_mul_req+s.phy_total_write_req+s.phy_write_mul_req+s.redo_writes) / s.seconds, 2) "IOPS/s",
       (s.phy_read_total_bytes+s.phy_write_total_bytes)/1024/1024 "throughput",
       round((s.phy_read_total_bytes+s.phy_write_total_bytes)/1024/1024 / s.seconds, 2) "throughput(MB)/s"
  from (select curr_phy_total_read_req - phy_total_read_req phy_total_read_req,
               curr_phy_read_mul_reqs - phy_read_mul_req phy_read_mul_req,
               curr_phy_total_write_req - phy_total_write_req phy_total_write_req,
               curr_phy_write_mul_req - phy_write_mul_req phy_write_mul_req,
               curr_redo_writes - redo_writes redo_writes,
               curr_phy_read_total_bytes - phy_read_total_bytes phy_read_total_bytes,
               curr_phy_write_total_bytes - phy_write_total_bytes phy_write_total_bytes,
               curr_redo_size - redo_size redo_size,
                round(((currtime + 0) - (lasttime + 0)) * 3600 * 24, 0) seconds,
               to_char(currtime, 'yy/mm/dd') snap_date,
               to_char(currtime, 'hh24:mi') currtime,
               currsnap_id endsnap_id,
               to_char(startup_time, 'yyyy-mm-dd hh24:mi:ss') startup_time,
               instance_number
          from (select a.instance_number instance_number,
                       a.phy_total_read_req phy_total_read_req,
                       a.phy_read_mul_req phy_read_mul_req,
                       a.phy_total_write_req phy_total_write_req,
                       a.phy_write_mul_req phy_write_mul_req,
                       a.redo_writes redo_writes,
                       a.phy_read_total_bytes phy_read_total_bytes,
                       a.phy_write_total_bytes phy_write_total_bytes,
                       a.redo_size redo_size,
                       lead(a.phy_total_read_req, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_phy_total_read_req,
                       lead(a.phy_read_mul_req, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_phy_read_mul_reqs,
                       lead(a.phy_total_write_req, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_phy_total_write_req,
                       lead(a.phy_write_mul_req, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_phy_write_mul_req,
                       lead(a.redo_writes, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_redo_writes,
                       lead(a.phy_read_total_bytes, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_phy_read_total_bytes,
                       lead(a.phy_write_total_bytes, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_phy_write_total_bytes,
                       lead(a.redo_size, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_redo_size,
                       b.end_interval_time lasttime,
                       lead(b.end_interval_time, 1, null) over(partition by b.startup_time order by b.end_interval_time) currtime,
                       lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) currsnap_id,
                       b.startup_time
                  from (select snap_id,
                               dbid,
                               instance_number,
                               sum(decode(stat_name, 'physical read total IO requests', value, 0)) phy_total_read_req,
                               sum(decode(stat_name,'physical read total multi block requests',value,0)) phy_read_mul_req,
                               sum(decode(stat_name, 'physical write total IO requests', value, 0)) phy_total_write_req,
                               sum(decode(stat_name, 'physical write total multi block requests', value, 0)) phy_write_mul_req,
                               sum(decode(stat_name, 'redo writes', value, 0)) redo_writes,
                               sum(decode(stat_name, 'physical read total bytes', value, 0)) phy_read_total_bytes,
                               sum(decode(stat_name, 'physical write total bytes', value, 0)) phy_write_total_bytes,
                               sum(decode(stat_name, 'redo size', value, 0)) redo_size     
                          from dba_hist_sysstat
                         where stat_name in
                               ('physical read total IO requests',
                               'physical read total multi block requests',
                               'physical write total IO requests',
                               'physical write total multi block requests',
                               'redo writes',
                               'physical read total bytes',
                               'physical write total bytes',
                               'redo size')
                         group by snap_id, dbid, instance_number) a,
                       dba_hist_snapshot b
                 where a.snap_id = b.snap_id
                   and a.dbid = b.dbid
                   and a.instance_number = b.instance_number
                 order by end_interval_time)) s,
       (select lead(a.value, 1, null) over(partition by b.startup_time order by b.end_interval_time) - a.value db_time,
               lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) endsnap_id,
               a.instance_number instance_number
          from dba_hist_sys_time_model a, dba_hist_snapshot b
         where a.snap_id = b.snap_id
           and a.dbid = b.dbid
           and a.instance_number = b.instance_number
           and a.stat_name = 'DB time'
           ) t
 where s.endsnap_id = t.endsnap_id
 and s.instance_number = t.instance_number
 and rownum <10
 order by "IOPS","throughput(MB)/s" desc;ca

prompt <p>数据访问指标
select metric_name,sum(case metric_name when 'DB Time Per Second' then value
                                        when 'Logical Reads Per Sec' then value
                                        when 'CPU Time Per User Call' then value
                                        when 'User Calls Per Sec' then value
                       else 0 end)
from v$metric where metric_name  in ('DB Time Per Second','Logical Reads Per Sec','CPU Time Per User Call','User Calls Per Sec')
and round(TO_NUMBER(END_TIME - BEGIN_TIME) * 24 * 60 * 60,0) =60 and group_id in (2,6)
group by metric_name;

prompt <p>客户端响应流程
select  
       s.instance_number,
       s.snap_date,
       to_char(round(s.seconds/60,2)) "elapse(min)",
       round(t.db_time / 1000000 / 60, 2) "DB time(min)",
       byte_send_net_client byte_send_net_client,
       byte_recv_net_client byte_recv_net_client,
       roundtrip_net roundtrip_net,
       round((byte_send_net_client+byte_recv_net_client) / roundtrip_net, 2) "bytes received per SQL*Net roundtrips"
  from (select curr_byte_send_net_client - byte_send_net_client byte_send_net_client,
               curr_byte_recv_net_client - byte_recv_net_client byte_recv_net_client,
               curr_roundtrip_net - roundtrip_net roundtrip_net,
                round(((currtime + 0) - (lasttime + 0)) * 3600 * 24, 0) seconds,
               to_char(currtime, 'yy/mm/dd') snap_date,
               to_char(currtime, 'hh24:mi') currtime,
               currsnap_id endsnap_id,
               to_char(startup_time, 'yyyy-mm-dd hh24:mi:ss') startup_time,
               instance_number
          from (select a.instance_number instance_number,
                       a.byte_send_net_client byte_send_net_client,
                       a.byte_recv_net_client byte_recv_net_client,
                       a.roundtrip_net roundtrip_net,
                       lead(a.byte_send_net_client, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_byte_send_net_client,
                       lead(a.byte_recv_net_client, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_byte_recv_net_client,
                       lead(a.roundtrip_net, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_roundtrip_net,
                       b.end_interval_time lasttime,
                       lead(b.end_interval_time, 1, null) over(partition by b.startup_time order by b.end_interval_time) currtime,
                       lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) currsnap_id,
                       b.startup_time
                  from (select snap_id,
                               dbid,
                               instance_number,
                               sum(decode(stat_name, 'bytes sent via SQL*Net to client', value, 0)) byte_send_net_client,
                               sum(decode(stat_name, 'bytes received via SQL*Net from client',value,0)) byte_recv_net_client,
                               sum(decode(stat_name, 'SQL*Net roundtrips to/from client', value, 0)) roundtrip_net
                          from dba_hist_sysstat
                         where stat_name in
                               ('bytes sent via SQL*Net to client',
                                'bytes received via SQL*Net from client',
                                'SQL*Net roundtrips to/from client')
                         group by snap_id, dbid, instance_number) a,
                       dba_hist_snapshot b
                 where a.snap_id = b.snap_id
                   and a.dbid = b.dbid
                   and a.instance_number = b.instance_number
                 order by end_interval_time)) s,
       (select lead(a.value, 1, null) over(partition by b.startup_time order by b.end_interval_time) - a.value db_time,
               lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) endsnap_id,
               a.instance_number instance_number
          from dba_hist_sys_time_model a, dba_hist_snapshot b
         where a.snap_id = b.snap_id
           and a.dbid = b.dbid
           and a.instance_number = b.instance_number
           and a.stat_name = 'DB time'
           ) t
 where s.endsnap_id = t.endsnap_id
 and s.instance_number = t.instance_number
 order by "bytes received per SQL*Net roundtrips"desc;