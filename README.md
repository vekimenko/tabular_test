**Test description**

This repository provides a quick prototype for a structured data classification Tensorflow model.  
  
A raw dataset is synthetically generated to reproduce the original log files and contains the following fields:  
- request_time  
- request_date  
- source  
- asn_number  
- client_ip  
- user_agent  
- device_name  
- request_host  
- request_path  
- http_status_code  
- total_bytes  
  
  
The view below prepares the dataset for preprocessing and training:  
  
```code

DROP VIEW IF EXISTS `strategy-bi-ltd.ml.vera_frames`;

CREATE VIEW `strategy-bi-ltd.ml.vera_frames` AS 
WITH tab AS (
	SELECT 
		FLOOR(TIME_DIFF(TIME(request_time), TIME "00:00:00", MINUTE)/15) frame,
		date_start,
		TIME(request_time) AS request_id,
		'source' AS SOURCE,
		asn AS asn_number,
		client_ip,
		user_agent,
		custom_field.device_name device_name,
		request_host,
		request_path,
		http_status_code,
		total_bytes,
		avg(turnaround_time)over() AS avg_turnaround_time,
		avg(transfer_time)over() AS avg_transfer_time
	FROM `strategy-bi-ltd.de_cdn_logs.de_akamai_linear`
	WHERE EXTRACT(YEAR from date_start) = 2024 AND EXTRACT(MONTH from date_start) = 2	
),
tab2 AS (
	SELECT 
		frame,
		MAX(date_start) AS date_start,
		MAX(asn_number) AS asn_number,
		MAX(client_ip) AS client_ip,
		MAX(user_agent) AS user_agent,
		MAX(device_name) AS device_name,
		MAX(request_host) AS request_host,
		MAX(request_path) AS request_path,
		MAX(http_status_code) AS http_status_code,
		round(sum(total_bytes)/1048576,2) total_MB_consumed,
		count(distinct request_id) cnt_request_id,
		MAX(avg_turnaround_time) AS avg_turnaround_time,
		MAX(avg_transfer_time) AS avg_transfer_time
	FROM tab
	GROUP BY frame )
SELECT 
	date_start,
	tab2.asn_number,
	client_ip,
	tab2.user_agent,
	replace(replace(replace(tab2.user_agent,'%20', ' '),'%5b', ''),'%5d','')  as user_agent2,
	device_name,
	request_host,
	request_path,
	REGEXP_EXTRACT(request_path, r'channel\((.*)\)') AS channel,
	http_status_code,
	avg_turnaround_time,
	avg_transfer_time,
	MIN(total_MB_consumed) AS min_total_MB_consumed,
	MAX(total_MB_consumed) AS max_total_MB_consumed,
	AVG(total_MB_consumed) AS avg_total_MB_consumed,
	COUNT(total_MB_consumed) AS cnt_total_MB_consumed,
	SUM(total_MB_consumed) AS sum_total_MB_consumed,
	MIN(cnt_request_id) AS min_cnt_request_id,
	MAX(cnt_request_id) AS max_cnt_request_id,
	AVG(cnt_request_id) AS avg_cnt_request_id,
	COUNT(cnt_request_id) AS cnt_cnt_request_id,
	SUM(cnt_request_id) AS sum_cnt_request_id,
	MAX(ulu.status) AS user_agent_status, 
	MAX(alu.asn_name) AS asn_name, 
	MAX(alu.asn_country) AS asn_country, 
	MAX(alu.asn_type) AS asn_type
FROM tab2
LEFT JOIN `strategy-bi-ltd.reference.user_agent_status` ulu on tab2.user_agent = ulu.user_agent
LEFT JOIN `strategy-bi-ltd.reference.asnlookup` alu on tab2.asn_number = alu.asn_number
GROUP BY 	  	
	date_start,
	tab2.asn_number,
	client_ip,
	tab2.user_agent,
	device_name,
	request_host,
	request_path,
	http_status_code,
	avg_turnaround_time,
	avg_transfer_time
;

```

The pipeline can be found in the notebook.  
  
The script of the pipeline can be found in baseline_module.py.  
  

 





