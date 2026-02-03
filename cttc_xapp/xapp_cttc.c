// xapp_ebrahim
#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/time_now_us.h"
#include "../../../../src/util/e2ap_ngran_types.h"
#include "../../../../src/util/alg_ds/ds/lock_guard/lock_guard.h"
#include "../../../../src/sm/kpm_sm/kpm_sm_id_wrapper.h"
#include "../../../../src/sm/rc_sm/rc_sm_id.h"
#include "../../../../src/sm/rc_sm/ie/rc_data_ie.h"
#include "common.c"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <librdkafka/rdkafka.h>
#include <glib.h>
#include <string.h>
#include <assert.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/ip_icmp.h>
#include <sys/time.h>
#include <errno.h>
#include <jansson.h>

#define PING_PKT_SIZE 64

// Structure to store Kafka configuration parameters
typedef struct {
    char bootstrap_servers[256];
    char topic[256];
    char client_id[256];
    char group_id[256];
    char ping_ip[256];
    // Add other necessary parameters as needed
} kafka_config;

// Define a structure to represent the "Kafka class"
typedef struct {
    rd_kafka_t *producer;  // Kafka producer
    rd_kafka_conf_t *conf; // Kafka configuration
    char errstr[512];      // Buffer for error messages
} Kafka;

// Declare the global Kafka instance
Kafka kafka_instance;
kafka_config kafkaConf; 

static
pthread_mutex_t mtx;

// ICMP header structure
struct icmp_hdr {
    u_int8_t type;
    u_int8_t code;
    u_int16_t checksum;
    u_int16_t id;
    u_int16_t sequence;
};

// Calculate checksum for the ICMP header
unsigned short checksum(void *b, int len) {
    unsigned short *buf = b;
    unsigned int sum = 0;
    unsigned short result;

    for (sum = 0; len > 1; len -= 2)
        sum += *buf++;
    if (len == 1)
        sum += *(unsigned char*)buf;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    result = ~sum;
    return result;
}

// Function to measure the time difference in milliseconds
double time_diff(struct timeval *start, struct timeval *end) {
    return (double)((end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0);
}

// Function to get the current timestamp
double get_current_time() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; // Convert to seconds with nanoseconds
}

// Function to ping an IP address and return the RTT
double ping(const char *ip_addr) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(ip_addr);

    if (addr.sin_addr.s_addr == INADDR_NONE) {
        fprintf(stderr, "Invalid IP address: %s\n", ip_addr);
        return -1;
    }

    // Create a raw socket
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sock < 0) {
        perror("socket");
        return -1;
    }

    struct icmp_hdr icmp_req;
    memset(&icmp_req, 0, sizeof(icmp_req));

    // Set ICMP Echo Request header
    icmp_req.type = ICMP_ECHO;
    icmp_req.code = 0;
    icmp_req.id = getpid();
    icmp_req.sequence = 1;
    icmp_req.checksum = checksum(&icmp_req, sizeof(icmp_req));

    struct timeval start_time, end_time;

    // Get the start time
    gettimeofday(&start_time, NULL);

    // Send the ping
    ssize_t bytes_sent = sendto(sock, &icmp_req, sizeof(icmp_req), 0, (struct sockaddr *)&addr, sizeof(addr));
    if (bytes_sent < 0) {
        perror("sendto");
        close(sock);
        return -1;
    } else {
        printf("Sent %ld bytes to %s\n", bytes_sent, ip_addr);
    }

    // Wait for the reply
    char buffer[PING_PKT_SIZE];
    socklen_t addr_len = sizeof(addr);

    ssize_t bytes_received = recvfrom(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&addr, &addr_len);
    if (bytes_received < 0) {
        perror("recvfrom");
        close(sock);
        return -1;
    } else {
        printf("Received %ld bytes from %s\n", bytes_received, ip_addr);
    }

    // Get the end time
    gettimeofday(&end_time, NULL);

    // Calculate and return the round-trip time (RTT)
    double rtt = time_diff(&start_time, &end_time);
    close(sock);
    return rtt;
}


#define MAX_LINE_LENGTH 256
#define ARR_SIZE(arr) ( sizeof((arr)) / sizeof((arr[0])) )



////////////
// Get RC Indication Messages -> begin
////////////

static void sm_cb_rc(sm_ag_if_rd_t const *rd, global_e2_node_id_t const* e2_node)
{
  assert(rd != NULL);
  assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
  assert(rd->ind.type == RAN_CTRL_STATS_V1_03);
  (void) e2_node;

  // Reading Indication Message Format 2
  e2sm_rc_ind_msg_frmt_2_t const *msg_frm_2 = &rd->ind.rc.ind.msg.frmt_2;

  printf("RC REPORT Style 2 - Call Process Outcome\n");

  // Sequence of UE Identifier
  //[1-65535]
  for (size_t i = 0; i < msg_frm_2->sz_seq_ue_id; i++)
  {
    // UE ID
    // Mandatory
    // 9.3.10
    switch (msg_frm_2->seq_ue_id[i].ue_id.type)
    {
      case GNB_UE_ID_E2SM:
        printf("UE connected to gNB with amf_ue_ngap_id = %lu\n", msg_frm_2->seq_ue_id[i].ue_id.gnb.amf_ue_ngap_id);
        break;
      default:
        printf("Not yet implemented UE ID type\n");
    }
  }
}

////////////
// Get RC Indication Messages -> end
////////////

static
void sm_cb_kpm(sm_ag_if_rd_t const* rd, 
               global_e2_node_id_t const* e2_node
               )
{
  assert(rd != NULL);
  assert(rd->type == INDICATION_MSG_AGENT_IF_ANS_V0);
  assert(rd->ind.type == KPM_STATS_V3_0);

  kpm_ind_data_t const* kpm = &rd->ind.kpm.ind;
  kpm_ric_ind_hdr_format_1_t const* hdr_frm_1 = &kpm->hdr.kpm_ric_ind_hdr_format_1;

  int64_t now = time_now_us();

  // define json
  char key[20];
  json_t *json_obj = json_object();
  if (!json_obj) {
      fprintf(stderr, "Error creating JSON object\n");
      return;
  }
  

  {
    lock_guard(&mtx);

#if defined(KPM_V2_01) || defined (KPM_V2_03)
    // collectStartTime (32bit) unit is second
    printf("KPM-v2 ind_msg latency > %ld s (minimum time unit is in second) from E2-node type %d ID %d\n",
           now/1000000 - hdr_frm_1->collectStartTime,
           e2_node->type, e2_node->nb_id.nb_id);
#elif defined(KPM_V3_00)
    // collectStartTime (64bit) unit is micro-second
    printf("KPM-v3 ind_msg latency = %ld μs from E2-node type %d ID %d\n",
           now - hdr_frm_1->collectStartTime,
           e2_node->type, e2_node->nb_id.nb_id);
#else
    static_assert(0!=0, "Unknown KPM version");
#endif

    if (kpm->msg.type == FORMAT_1_INDICATION_MESSAGE) {
      kpm_ind_msg_format_1_t const* msg_frm_1 = &kpm->msg.frm_1;
      for (size_t i = 0; i < msg_frm_1->meas_data_lst_len; i++) {
        for (size_t j = 0; j < msg_frm_1->meas_data_lst[i].meas_record_len; j++) {
          if (msg_frm_1->meas_data_lst[i].meas_record_lst[j].value == INTEGER_MEAS_VALUE)
            printf("meas record INTEGER_MEAS_VALUE value %d\n",msg_frm_1->meas_data_lst[i].meas_record_lst[j].int_val);
          else if (msg_frm_1->meas_data_lst[i].meas_record_lst[j].value == REAL_MEAS_VALUE)
            printf("meas record REAL_MEAS_VALUE value %f\n", msg_frm_1->meas_data_lst[i].meas_record_lst[j].real_val);
          else
            printf("meas record NO_VALUE_MEAS_VALUE value\n");
        }
      }
    } else if (kpm->msg.type == FORMAT_3_INDICATION_MESSAGE) {
      kpm_ind_msg_format_3_t const* msg_frm_3 = &kpm->msg.frm_3;
      // Reported list of measurements per UE
      json_object_set_new(json_obj, "UEs_number", json_integer((int64_t)msg_frm_3->ue_meas_report_lst_len));

      // Calculate latency once per message (if applicable)
      double rtt = ping(kafkaConf.ping_ip);
      if (rtt >= 0) {
          //kafka_send_key_value("latancy", (double)rtt);
          json_object_set_new(json_obj, "latency", json_real((double)rtt));
      } else {
          printf("Failed to ping %s\n", kafkaConf.ping_ip);
      }

      for (size_t i = 0; i < msg_frm_3->ue_meas_report_lst_len; i++) {
        json_t *ue_obj = json_object(); // Create a new JSON object for this UE
        char ue_key[20];
        sprintf(ue_key, "UE%ld", i + 1); // Generates UE1, UE2, etc.

        // Process UE ID
        switch (msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.type)
        {
          case GNB_UE_ID_E2SM:
            if (msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb.gnb_cu_ue_f1ap_lst != NULL) {
              for (size_t j = 0; j < msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb.gnb_cu_ue_f1ap_lst_len; j++)
                printf("UE ID type = gNB-CU, gnb_cu_ue_f1ap = %u\n", msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb.gnb_cu_ue_f1ap_lst[j]);
            } else {
              printf("UE ID type = gNB, amf_ue_ngap_id = %lu\n", msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb.amf_ue_ngap_id);
            }
            break;

          case GNB_DU_UE_ID_E2SM:
            printf("UE ID type = gNB-DU, gnb_cu_ue_f1ap = %u\n", msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb_du.gnb_cu_ue_f1ap);
            int64_t ran_ue_id = (int64_t)msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb_du.ran_ue_id;
            json_object_set_new(ue_obj, "ID", json_integer(i));
            break;
          case GNB_CU_UP_UE_ID_E2SM:
            printf("UE ID type = gNB-CU, gnb_cu_cp_ue_e1ap = %u\n", msg_frm_3->meas_report_per_ue[i].ue_meas_report_lst.gnb_cu_up.gnb_cu_cp_ue_e1ap);
            break;

          default:
            assert(false && "UE ID type not yet implemented");
        }
        kpm_ind_msg_format_1_t const* msg_frm_1 = &msg_frm_3->meas_report_per_ue[i].ind_msg_format_1;

        // UE Measurements per granularity period
        for (size_t j = 0; j<msg_frm_1->meas_data_lst_len; j++) {
          
          for (size_t z = 0; z<msg_frm_1->meas_data_lst[j].meas_record_len; z++) {
            if (msg_frm_1->meas_info_lst_len > 0) {
              switch (msg_frm_1->meas_info_lst[z].meas_type.type) {
                case NAME_MEAS_TYPE:
                {
                  // Get the Measurement Name
                  char meas_info_name_str[msg_frm_1->meas_info_lst[z].meas_type.name.len + 1];
                  memcpy(meas_info_name_str, msg_frm_1->meas_info_lst[z].meas_type.name.buf, msg_frm_1->meas_info_lst[z].meas_type.name.len);
                  meas_info_name_str[msg_frm_1->meas_info_lst[z].meas_type.name.len] = '\0';

                  // Get the value of the Measurement
                  switch (msg_frm_1->meas_data_lst[j].meas_record_lst[z].value)
                  {
                    case REAL_MEAS_VALUE:
                      printf("%s = %.2f\n", meas_info_name_str, msg_frm_1->meas_data_lst[j].meas_record_lst[z].real_val);
                      if (meas_info_name_str != NULL && strlen(meas_info_name_str) > 0) {
                        //kafka_send_key_value(meas_info_name_str, (double)msg_frm_1->meas_data_lst[j].meas_record_lst[z].real_val);
                        json_object_set_new(ue_obj, meas_info_name_str, json_real(msg_frm_1->meas_data_lst[j].meas_record_lst[z].real_val));
                      } else {
                        printf("Invalid Measurement Name String\n");
                      }
                      break;

                    case INTEGER_MEAS_VALUE:
                      printf("%s = %d\n", meas_info_name_str, msg_frm_1->meas_data_lst[j].meas_record_lst[z].int_val);
                      if (meas_info_name_str != NULL && strlen(meas_info_name_str) > 0) {
                        //kafka_send_key_value(meas_info_name_str, (double)msg_frm_1->meas_data_lst[j].meas_record_lst[z].int_val);
                        json_object_set_new(ue_obj, meas_info_name_str, json_integer(msg_frm_1->meas_data_lst[j].meas_record_lst[z].int_val));
                      } else {
                        printf("Invalid Measurement Name String\n");
                      }
                      break;

                    default:
                      assert("Value not recognized");
                  }
                  break;
                }

                default:
                  assert(false && "Measurement Type not yet implemented");
              }
            }
            if (msg_frm_1->meas_data_lst[j].incomplete_flag && *msg_frm_1->meas_data_lst[j].incomplete_flag == TRUE_ENUM_VALUE)
              printf("Measurement Record not reliable\n");
          }
        }
        json_object_set_new(json_obj, ue_key, ue_obj);
      }
    
    } else {
      printf("unknown kpm ind format\n");
    }

  }
  // get current time
  double current_time = get_current_time();
  json_object_set_new(json_obj, "time", json_real(current_time));
  //kafka_send_key_value("time", (double)current_time);
  // ping latance
  kafka_send_json_message(json_obj, key);
}

static e2sm_rc_ev_trg_frmt_2_t gen_rc_ev_trig_frm_2(void)
{
  e2sm_rc_ev_trg_frmt_2_t ev_trigger = {0};

  //  Call Process Type ID
  //  Mandatory
  //  9.3.15
  ev_trigger.call_proc_type_id = 3; // Mobility Management

  // Call Breakpoint ID
  // Mandatory
  // 9.3.49
  ev_trigger.call_break_id = 1; // Handover Preparation

  // Associated E2 Node Info
  // Optional
  // 9.3.29
  ev_trigger.assoc_e2_node_info = NULL;

  // Associated UE Info
  // Optional
  // 9.3.26
  ev_trigger.assoc_ue_info = NULL;

  return ev_trigger;
}

static
e2sm_rc_event_trigger_t gen_rc_ev_trig(e2sm_rc_ev_trigger_format_e act_frm)
{
  e2sm_rc_event_trigger_t dst = {0};

  if (act_frm == FORMAT_2_E2SM_RC_EV_TRIGGER_FORMAT) {
    dst.format = FORMAT_2_E2SM_RC_EV_TRIGGER_FORMAT;
    dst.frmt_2 = gen_rc_ev_trig_frm_2();
  } else {
    assert(0!=0 && "not support event trigger type");
  }

  return dst;
}

static
kpm_event_trigger_def_t gen_kpm_ev_trig(uint64_t period)
{
  kpm_event_trigger_def_t dst = {0};

  dst.type = FORMAT_1_RIC_EVENT_TRIGGER;
  dst.kpm_ric_event_trigger_format_1.report_period_ms = period;

  return dst;
}

static
meas_info_format_1_lst_t gen_meas_info_format_1_lst(const act_name_id_t act)
{
  meas_info_format_1_lst_t dst = {0};

  // use id
  if (!strcasecmp(act.name, "null")) {
    dst.meas_type.type = ID_MEAS_TYPE;
    dst.meas_type.id = act.id;
  } else { // use name
    dst.meas_type.type = NAME_MEAS_TYPE;
    // ETSI TS 128 552
    dst.meas_type.name = cp_str_to_ba(act.name);
  }

  dst.label_info_lst_len = 1;
  dst.label_info_lst = calloc(1, sizeof(label_info_lst_t));
  assert(dst.label_info_lst != NULL && "Memory exhausted");

  // No Label
  dst.label_info_lst[0].noLabel = calloc(1, sizeof(enum_value_e));
  assert(dst.label_info_lst[0].noLabel != NULL && "Memory exhausted");
  *dst.label_info_lst[0].noLabel = TRUE_ENUM_VALUE;

  return dst;
}

static
kpm_act_def_format_1_t gen_kpm_act_def_frmt_1(const sub_oran_sm_t sub_sm, uint32_t period_ms)
{
  kpm_act_def_format_1_t dst = {0};

  dst.gran_period_ms = period_ms;

  dst.meas_info_lst_len = sub_sm.act_len;
  dst.meas_info_lst = calloc(dst.meas_info_lst_len, sizeof(meas_info_format_1_lst_t));
  assert(dst.meas_info_lst != NULL && "Memory exhausted");

  for(size_t i = 0; i < dst.meas_info_lst_len; i++) {
    dst.meas_info_lst[i] = gen_meas_info_format_1_lst(sub_sm.actions[i]);
  }

  return dst;
}

static
kpm_act_def_format_4_t gen_kpm_act_def_frmt_4(const sub_oran_sm_t sub_sm, uint32_t period_ms)
{
  kpm_act_def_format_4_t dst = {0};

  // [1, 32768]
  dst.matching_cond_lst_len = 1;

  dst.matching_cond_lst = calloc(dst.matching_cond_lst_len, sizeof(matching_condition_format_4_lst_t));
  assert(dst.matching_cond_lst != NULL && "Memory exhausted");

  test_info_lst_t* test_info_lst = &dst.matching_cond_lst[0].test_info_lst;
  test_info_lst->test_cond_type = S_NSSAI_TEST_COND_TYPE;
  test_info_lst->S_NSSAI = TRUE_TEST_COND_TYPE;

  test_cond_e* test_cond = calloc(1, sizeof(test_cond_e));
  assert(test_cond != NULL && "Memory exhausted");
  *test_cond = EQUAL_TEST_COND;
  test_info_lst->test_cond = test_cond;

  test_cond_value_t* test_cond_value = calloc(1, sizeof(test_cond_value_t));
  assert(test_cond_value != NULL && "Memory exhausted");
  test_cond_value->type = INTEGER_TEST_COND_VALUE;
  test_cond_value->int_value = calloc(1, sizeof(int64_t));
  assert(test_cond_value->int_value != NULL && "Memory exhausted");
  *test_cond_value->int_value = 1;
  test_info_lst->test_cond_value = test_cond_value;

  // Action definition Format 1
  dst.action_def_format_1 = gen_kpm_act_def_frmt_1(sub_sm, period_ms);  // 8.2.1.2.1

  return dst;
}

static
e2sm_rc_act_def_frmt_1_t gen_rc_act_def_frm_1(const sub_oran_sm_t sub_sm)
{
  e2sm_rc_act_def_frmt_1_t act_def_frm_1 = {0};

  // Parameters to be Reported List
  // [1-65535]
  // 8.2.2
  act_def_frm_1.sz_param_report_def = sub_sm.act_len;
  act_def_frm_1.param_report_def = calloc(act_def_frm_1.sz_param_report_def, sizeof(param_report_def_t));
  assert(act_def_frm_1.param_report_def != NULL && "Memory exhausted");

  // Current UE ID RAN Parameter
  for (size_t i = 0; i < act_def_frm_1.sz_param_report_def; i++) {
    // use id
    if (!strcasecmp(sub_sm.actions[i].name, "null")) {
      act_def_frm_1.param_report_def[i].ran_param_id = sub_sm.actions[i].id;
    } else { // use name
      assert(0!=0 && "not supported Name for RC action definition\n");
    }
  }

  return act_def_frm_1;
}

static
e2sm_rc_action_def_t gen_rc_act_def(const sub_oran_sm_t sub_sm, uint32_t ric_style_type, e2sm_rc_act_def_format_e act_frmt)
{
  e2sm_rc_action_def_t dst = {0};
  dst.ric_style_type = ric_style_type;
  dst.format = act_frmt;
  if (act_frmt == FORMAT_1_E2SM_RC_ACT_DEF) {
    dst.frmt_1 = gen_rc_act_def_frm_1(sub_sm);
  } else {
    assert(0!=0 && "not supported RC action definition\n");
  }

  return dst;
}

static
kpm_act_def_t gen_kpm_act_def(const sub_oran_sm_t sub_sm, format_action_def_e act_frm, uint32_t period_ms)
{
  kpm_act_def_t dst = {0};

  if (act_frm == FORMAT_1_ACTION_DEFINITION) {
    dst.type = FORMAT_1_ACTION_DEFINITION;
    dst.frm_1 = gen_kpm_act_def_frmt_1(sub_sm, period_ms);
  } else if (act_frm == FORMAT_4_ACTION_DEFINITION) {
    dst.type = FORMAT_4_ACTION_DEFINITION;
    dst.frm_4 = gen_kpm_act_def_frmt_4(sub_sm, period_ms);
  } else {
    assert(0!=0 && "not support action definition type");
  }

  return dst;
}

void kafka_send_json_message(json_t *json_obj, const char *key) {
    // Create a JSON object
    if (!json_obj) {
        fprintf(stderr, "Error creating JSON object\n");
        return;
    }

    // Serialize the JSON object to a string
    char *json_string = json_dumps(json_obj, 0);
    if (!json_string) {
        fprintf(stderr, "Error serializing JSON object\n");
        json_decref(json_obj);
        return;
    }

    // Sending the message to Kafka
    rd_kafka_topic_t *topic;
    rd_kafka_resp_err_t err;

    if (kafkaConf.topic == NULL || strlen(kafkaConf.topic) == 0) {
        fprintf(stderr, "Kafka topic is not set.\n");
        json_decref(json_obj);
        free(json_string);
        return;
    }

    // Produce the message with both key and value
    err = rd_kafka_producev(kafka_instance.producer,
                            RD_KAFKA_V_TOPIC(kafkaConf.topic),
                            RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                            RD_KAFKA_V_KEY((void*)key, strlen(key)),
                            RD_KAFKA_V_VALUE(json_string, strlen(json_string)),
                            RD_KAFKA_V_OPAQUE(NULL),
                            RD_KAFKA_V_END);

    if (err) {
        g_error("Failed to produce to topic %s: %s", kafkaConf.topic, rd_kafka_err2str(err));
        return;
    } else {
        g_message("Produced event to topic %s: key = %12s", kafkaConf.topic, key);
    }

    // Clean up
    json_decref(json_obj);
    free(json_string);

}

// Function to send a key-value pair with a numeric key and value
void kafka_send_key_value(const char * key, double value) {
    printf("Kafka send: key=%s, value=%.2f\n", key, value);
    rd_kafka_topic_t *topic;
    rd_kafka_resp_err_t err;

    if (kafkaConf.topic == NULL || strlen(kafkaConf.topic) == 0) 
    {
      fprintf(stderr, "Kafka topic is not set.\n");
      return;
    }
  
    // Serialize key and value
    char value_str[64];  // Sufficient space for a double string representation
    snprintf(value_str, sizeof(value_str), "%.12f", value);
    // Produce the message with both key and value
    err = rd_kafka_producev(kafka_instance.producer,
                            RD_KAFKA_V_TOPIC(kafkaConf.topic),
                            RD_KAFKA_V_MSGFLAGS(RD_KAFKA_MSG_F_COPY),
                            RD_KAFKA_V_KEY((void*)key, strlen(key)),
                            RD_KAFKA_V_VALUE((void*)value_str, strlen(value_str)),
                            RD_KAFKA_V_OPAQUE(NULL),
                            RD_KAFKA_V_END);

    if (err) {
        g_error("Failed to produce to topic %s: %s", kafkaConf.topic, rd_kafka_err2str(err));
        return;
    } else {
        g_message("Produced event to topic %s: key = %12s ", kafkaConf.topic, key);
    }

    rd_kafka_poll(kafka_instance.producer, 0);
}

/* Optional per-message delivery callback (triggered by poll() or flush())
 * when a message has been successfully delivered or permanently
 * failed delivery (after retries).
 */
static
void dr_msg_cb (rd_kafka_t *kafka_handle, const rd_kafka_message_t *rkmessage, void *opaque) 
{
    if (rkmessage->err) {
        g_error("Message delivery failed: %s", rd_kafka_err2str(rkmessage->err));
    }
}

// Function to initialize the Kafka producer
void kafka_init() {
    kafka_instance.conf = rd_kafka_conf_new();
    
    kafka_set_config(kafka_instance.conf, "bootstrap.servers", kafkaConf.bootstrap_servers);
    // Install a delivery-error callback.
    rd_kafka_conf_set_dr_msg_cb(kafka_instance.conf, dr_msg_cb);
    // Create the Producer instance.
    kafka_instance.producer = rd_kafka_new(RD_KAFKA_PRODUCER, kafka_instance.conf, kafka_instance.errstr, sizeof(kafka_instance.errstr));
    if (!kafka_instance.producer) {
      g_error("Failed to create new producer: %s", kafka_instance.errstr);
      return;
    }
}

// Function to clean up and destroy the Kafka producer
void kafka_destroy() {
     // free kafka
  // Block until the messages are all sent.
  g_message("Flushing final messages..");
  rd_kafka_flush(kafka_instance.producer, 10 * 1000);

  if (rd_kafka_outq_len(kafka_instance.producer) > 0) {
      g_error("%d message(s) were not delivered", rd_kafka_outq_len(kafka_instance.producer));
  }
  rd_kafka_destroy(kafka_instance.producer);
}

// Function to parse a line in the format "key=value"
void parse_config_line(const char *line) {
    char key[MAX_LINE_LENGTH];
    char value[MAX_LINE_LENGTH];

    // Split the line into key and value
    if (sscanf(line, "%[^=]=%s", key, value) == 2) {
        printf("%s: %s\n", key, value);
        if (strcmp(key, "bootstrap.servers") == 0) {
            strncpy(kafkaConf.bootstrap_servers, value, sizeof(kafkaConf.bootstrap_servers));
        } else if (strcmp(key, "topic") == 0) {
            strncpy(kafkaConf.topic, value, sizeof(kafkaConf.topic));
        } else if (strcmp(key, "group.id") == 0) {
            strncpy(kafkaConf.group_id, value, sizeof(kafkaConf.group_id));
        } else if (strcmp(key, "client.id") == 0) {
          strncpy(kafkaConf.client_id, value, sizeof(kafkaConf.client_id));
        } else if (strcmp(key, "ping.ip") == 0) {
          strncpy(kafkaConf.ping_ip, value, sizeof(kafkaConf.ping_ip));
        }
        // Add more keys and corresponding values as needed
    }
}

// Function to read and parse the Kafka config file
void read_kafka_config(const char *filename) {
    printf("reading kakfka conf file ...");
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open config file %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file) != NULL) {
        // Remove trailing newline
        line[strcspn(line, "\n")] = '\0';

        // Ignore comments and empty lines
        if (strlen(line) > 0 && line[0] != '#') {
            parse_config_line(line);
        }
    }

    fclose(file);
}


void print_gnb_info(const e2_node_arr_xapp_t* nodes)
{
  assert(nodes != NULL);
  for (size_t i = 0; i < nodes->len; i++) {
    e2ap_ngran_node_t ran_type = nodes->n[i].id.type;
    if (E2AP_NODE_IS_MONOLITHIC(ran_type))
      printf("E2 node %ld info: nb_id %d, mcc %d, mnc %d, mnc_digit_len %d, ran_type %s\n",
             i,
             nodes->n[i].id.nb_id.nb_id,
             nodes->n[i].id.plmn.mcc,
             nodes->n[i].id.plmn.mnc,
             nodes->n[i].id.plmn.mnc_digit_len,
             get_e2ap_ngran_name(ran_type));
    else
      printf("E2 node %ld info: nb_id %d, mcc %d, mnc %d, mnc_digit_len %d, ran_type %s, cu_du_id %lu\n",
             i,
             nodes->n[i].id.nb_id.nb_id,
             nodes->n[i].id.plmn.mcc,
             nodes->n[i].id.plmn.mnc,
             nodes->n[i].id.plmn.mnc_digit_len,
             get_e2ap_ngran_name(ran_type),
             *nodes->n[i].id.cu_du_id);

    printf("E2 node %ld supported RAN function's IDs:", i);
    for (size_t j = 0; j < nodes->n[i].len_rf; j++)
      printf(", %d", nodes->n[i].rf[j].id);
    printf("\n");
  }
}


int main(int argc, char *argv[])
{
  int argc_c = 3;
  char *argv_c[3] = {NULL}; // Array to hold arguments for -c
  char *kafka_conf= NULL; // Array to hold arguments for -k
  int i;

  // Iterate over the argv array and classify options
  for (i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
      argv_c[0] = argv[0]; // program name
      argv_c[1] = argv[i]; // -c option
      argv_c[2] = argv[i + 1]; // config file for -c
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      kafka_conf = argv[i + 1]; // config file for -k
    }
  }

  fr_args_t args = init_fr_args(argc_c, argv_c);
  defer({ free_fr_args(&args); });
  // ################## KAFKA #########################3
  // ini kafka
  printf("Kafka Configuration:\n");
  read_kafka_config(kafka_conf);
  printf("  bootstrap.servers = %s\n", kafkaConf.bootstrap_servers);
  printf("  topic = %s\n", kafkaConf.topic);
  printf("  client.id = %s\n", kafkaConf.client_id);
  printf("  group.id = %s\n", kafkaConf.group_id);
  kafka_init();
  // ################## KAFKA #########################

  //Init the xApp
  init_xapp_api(&args);
  sleep(1);

  e2_node_arr_xapp_t nodes ;

  int retries = 5;  // Retry for up to 5 seconds (50 x 100 ms)
  int success = 0;

  while (retries-- > 0) {
      nodes = e2_nodes_xapp_api();
      if (nodes.len > 0) {
          success = 1;
          break;
      }
      free_e2_node_arr_xapp(&nodes);  // Free memory before the next retry
      sleep(5);  // 5-second delay
  }

  defer({ free_e2_node_arr_xapp(&nodes); });

  if (!success) {
      fprintf(stderr, "No connected E2 nodes found within the 5-second window.\n");
      exit(EXIT_FAILURE);
  }
  
  // = e2_nodes_xapp_api();
  // defer({ free_e2_node_arr_xapp(&nodes); });

  // assert(nodes.len > 0);
  printf("Connected E2 nodes = %d\n", nodes.len);
  print_gnb_info(&nodes);

  //Init SM handler
  sm_ans_xapp_t* kpm_handle = NULL;
  sm_ans_xapp_t* rc_handle = NULL;

  if(nodes.len > 0){
    kpm_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) );
    assert(kpm_handle  != NULL);
    rc_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) );
    assert(rc_handle  != NULL);
  }

  int n_kpm_handle = 0;
  int n_rc_handle = 0;

  //Subscribe SMs for all the E2-nodes
  for (int i = 0; i < nodes.len; i++) {
    e2_node_connected_xapp_t* n = &nodes.n[i];
    for (size_t j = 0; j < n->len_rf; j++)
      printf("Registered node %d ran func id = %d \n ", i, n->rf[j].id);

    for (int32_t j = 0; j < args.sub_oran_sm_len; j++) {
      if (!strcasecmp(args.sub_oran_sm[j].name, "kpm")) {
        kpm_sub_data_t kpm_sub = {0};
        defer({ free_kpm_sub_data(&kpm_sub); });

        // KPM Event Trigger
        uint64_t period_ms = args.sub_oran_sm[j].time;
        kpm_sub.ev_trg_def = gen_kpm_ev_trig(period_ms);
        printf("[xApp]: reporting period = %lu [ms]\n", period_ms);
        // KPM Action Definition
        kpm_sub.sz_ad = 1;
        kpm_sub.ad = calloc(1, sizeof(kpm_act_def_t));
        assert(kpm_sub.ad != NULL && "Memory exhausted");
        format_action_def_e act_type = END_ACTION_DEFINITION;
        if (args.sub_oran_sm[j].format == 1)
          act_type = FORMAT_1_ACTION_DEFINITION;
        else if (args.sub_oran_sm[j].format == 4)
          act_type = FORMAT_4_ACTION_DEFINITION;
        else
          assert(0!=0 && "not supported action definition format");

        *kpm_sub.ad = gen_kpm_act_def((const sub_oran_sm_t)args.sub_oran_sm[j], act_type, period_ms);
        // TODO: implement e2ap_ngran_eNB
        if (n->id.type == e2ap_ngran_eNB)
          continue;
        if (strcasecmp(args.sub_oran_sm[j].ran_type, get_e2ap_ngran_name(n->id.type)))
          continue;
        printf("xApp subscribes RAN Func ID %d in E2 node idx %d, nb_id %d\n", SM_KPM_ID, i, n->id.nb_id.nb_id);
        kpm_handle[i] = report_sm_xapp_api(&nodes.n[i].id, SM_KPM_ID, &kpm_sub, sm_cb_kpm);
        assert(kpm_handle[i].success == true);
        n_kpm_handle += 1;

      } else if (!strcasecmp(args.sub_oran_sm[j].name, "rc")) {
        rc_sub_data_t rc_sub = {0};
        defer({ free_rc_sub_data(&rc_sub); });

        // RC Event Trigger
        rc_sub.et = gen_rc_ev_trig(FORMAT_2_E2SM_RC_EV_TRIGGER_FORMAT);

        // RC Action Definition
        rc_sub.sz_ad = 1;
        rc_sub.ad = calloc(rc_sub.sz_ad, sizeof(e2sm_rc_action_def_t));
        assert(rc_sub.ad != NULL && "Memory exhausted");
        e2sm_rc_act_def_format_e act_type = END_E2SM_RC_ACT_DEF;
        if (args.sub_oran_sm[j].format == 1)
          act_type = FORMAT_1_E2SM_RC_ACT_DEF;
        else
          assert(0!=0 && "not supported action definition format");

        // use RIC style 2 by default
        *rc_sub.ad = gen_rc_act_def((const sub_oran_sm_t)args.sub_oran_sm[j], 2, act_type);

        // RC HO only supports for e2ap_ngran_gNB
        if (n->id.type == e2ap_ngran_eNB || n->id.type == e2ap_ngran_gNB_CU || n->id.type == e2ap_ngran_gNB_DU)
          continue;
        if (strcasecmp(args.sub_oran_sm[j].ran_type, get_e2ap_ngran_name(n->id.type)))
          continue;
        printf("xApp subscribes RAN Func ID %d in E2 node idx %d, nb_id %d\n", SM_RC_ID, i, n->id.nb_id.nb_id);
        rc_handle[i] = report_sm_xapp_api(&nodes.n[i].id, SM_RC_ID, &rc_sub, sm_cb_rc);
        assert(rc_handle[i].success == true);
        n_rc_handle += 1;

      } else {
        assert(0!=0 && "unknown SM in .conf");
      }
    }
  }

  char input;
  printf("Enter q to quit: ");

  while (1) {
    input = getchar();  // Read a single character from the user

    // Consume the newline character left in the buffer by getchar()
    while (getchar() != '\n');  

    if (input == 'q' || input == 'Q') {
        printf("Quitting the program.\n");
        break;
    } else {
        printf("You entered: %c\n", input);
    }
  }
  

  for(int i = 0; i < n_kpm_handle; ++i) {
    rm_report_sm_xapp_api(kpm_handle[i].u.handle);
    sleep(1);
  }

  for(int i = 0; i < n_rc_handle; ++i) {
    rm_report_sm_xapp_api(rc_handle[i].u.handle);
    sleep(1);
  }

  // free sm handel
  if(n_kpm_handle > 0) {
    free(kpm_handle);
  }
  if(n_rc_handle > 0) {
    free(rc_handle);
  }
  //Stop the xApp
  while(try_stop_xapp_api() == false)
   usleep(1000);

  kafka_destroy();

  printf("Test xApp run SUCCESSFULLY\n");
}


  // // Print the arguments classified for -c option
  // if (argv_c[0] != NULL) {
  //   printf("Arguments for -c option:\n");
  //   for (i = 0; i < 3; i++) {
  //     printf("argv_c[%d] = %s\n", i, argv_c[i]);
  //   }
  // }

  // printf("\n");

  // // Print the arguments classified for -k option
  // if (argv_k[0] != NULL) {
  //   printf("Arguments for -k option:\n");
  //   for (i = 0; i < 3; i++) {
  //       printf("argv_k[%d] = %s\n", i, argv_k[i]);
  //   }
  // }
