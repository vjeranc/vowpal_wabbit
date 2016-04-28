/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include "search_sequencetask.h"
#include "vw.h"
#include <sstream>

namespace SequenceTask         { Search::search_task task = { "sequence",          run, initialize, nullptr,   nullptr,  nullptr     }; }
namespace SequenceMSDTask         { Search::search_task task = { "sequence_msd",          run, initialize, finish,   nullptr,  nullptr     }; }
namespace SequenceSpanTask     { Search::search_task task = { "sequencespan",      run, initialize, finish, setup, takedown }; }
namespace SequenceTaskCostToGo { Search::search_task task = { "sequence_ctg",      run, initialize, nullptr,   nullptr,  nullptr     }; }
namespace ArgmaxTask           { Search::search_task task = { "argmax",            run, initialize, finish,   nullptr,  nullptr     }; }
namespace SequenceTask_DemoLDF { Search::search_task task = { "sequence_demoldf",  run, initialize, finish, nullptr,  nullptr     }; }

namespace SequenceTask
{
void initialize(Search::search& sch, size_t& /*num_actions*/, po::variables_map& /*vm*/)
{ sch.set_options( Search::AUTO_CONDITION_FEATURES  |    // automatically add history features to our examples, please
                   Search::AUTO_HAMMING_LOSS        |    // please just use hamming loss on individual predictions -- we won't declare loss
                   Search::EXAMPLES_DONT_CHANGE     |    // we don't do any internal example munging
                   0);
}

void run(Search::search& sch, vector<example*>& ec)
{ Search::predictor P(sch, (ptag)0);
  for (size_t i=0; i<ec.size(); i++)
  { action oracle     = ec[i]->l.multi.label;
    size_t prediction = P.set_tag((ptag)i+1).set_input(*ec[i]).set_oracle(oracle).set_condition_range((ptag)i, sch.get_history_length(), 'p').predict();

    if (sch.output().good())
      sch.output() << sch.pretty_label((uint32_t)prediction) << ' ';
  }
}
}

namespace SequenceMSDTask
{
using namespace Search;
struct tag {
  action a;
  ptag i;
};
struct task_data
{ string constraints;
  vector<vector<v_array<action>>> allowed_per_pos;
  v_array<action> allowed_on_zero;
  v_array<v_array<tag>> tags;
};

inline void parse_constraints(string &constraints, task_data * data) {
  istringstream ss(constraints);
  struct msd {
    uint32_t pos;
    uint32_t action;
    uint32_t position;
  } m;
  vector<msd> cs;
  char colon;
  while(ss >> m.pos >> colon >> m.action >> colon >> m.position >> colon) {
    cs.emplace_back(m);
  }
  std::sort(cs.begin(), cs.end(), [](msd & a, msd &b){
    return std::make_tuple(a.position) < std::make_tuple(b.position);
  });
  for(auto && c : cs) {
    if (c.position != 0) break;
    data->allowed_on_zero.push_back(c.action);
  }
  size_t poscount = data->allowed_on_zero.size();
  data->allowed_per_pos.resize(poscount+1); // not using allowed_per_pos[0]
  for(auto && c : cs) {
    if (c.position == 0) continue;
    vector<v_array<action>> & v = data->allowed_per_pos[c.pos];
    if (v.size() <= c.position) { // TODO not using v[0]
      v.resize(c.position+1);
    }
    v[c.position].push_back(c.action);
  }
  // int i = 0;
  // for(vector<v_array<action>> & v : data->allowed_per_pos) {
  //   cerr << i << ' ';
  //   for(v_array<action> &as : v) {
  //     for (action a : as) {
  //       cerr << a << ' ';
  //     }
  //     cerr << '|';
  //   }
  //   i++;
  //   cerr << '\n';
  // }
}

void initialize(Search::search& sch, size_t& /*num_actions*/, po::variables_map& vm)
{ vw& all = sch.get_vw_pointer_unsafe();
  task_data *data = new task_data();
  sch.set_task_data<task_data>(data);
  new_options(all, "MSD Tagger Options")
  ("constraints", po::value<string>(&(data->constraints)), "For each POS tag and for each position contains a positional constraint.");
  add_options(all);

  check_option<string>(data->constraints, all, vm, "constraints", false,
                       string_equal,
                       "warning: specified --constraints different than the one loaded from regressor. using loaded value of: ", "");
  sch.set_options( Search::AUTO_CONDITION_FEATURES  |
                   // Search::AUTO_HAMMING_LOSS        |
                   Search::EXAMPLES_DONT_CHANGE     |
                   0);
  sch.set_label_parser( COST_SENSITIVE::cs_label, [](polylabel&l) -> bool { return l.cs.costs.size() == 0; });
  if (!vm.count("constraints")) {
    THROW("--constraints need to be specified: ")
  }
  parse_constraints(data->constraints, data);
}

void finish(Search::search& sch)
{ task_data *data = sch.get_task_data<task_data>();
  for(v_array<tag> & v : data->tags) {
    v.delete_v();
  }
  data->tags.delete_v();
  delete data;
}

inline action get_label(example* ex, unsigned pass){
  v_array<COST_SENSITIVE::wclass>& costs = ex->l.cs.costs;
  return costs.size() > pass ? costs[pass].class_index : 0;
}

inline v_array<action> & get_allowed(task_data *data,size_t pass, size_t i) {
  if (pass == 0) { // pos tags on first pass
    return data->allowed_on_zero;
  }
  // positional tags on others depending on the previously predicted pos
  action pos = data->tags[i][0].a;
  return data->allowed_per_pos[pos][pass];
}

inline void resize_tags(task_data * data, size_t size) {
  data->tags.resize(size);
}

inline void push_tag(task_data * data, size_t i, ptag p, action prediction, size_t pass) {
  if (pass == 0) data->tags[i].erase();
  data->tags[i].push_back(tag{prediction, p});
}

inline ptag get_previous_ptag(task_data * data, size_t i) {
  if (i == 0) return -1;
  return data->tags[i-1][0].i;
}

inline bool has_positional_decision(task_data * data, size_t i, size_t pass) {
  if (pass == 0) return true;
  action pos = data->tags[i][0].a;
  size_t numOfPositions = data->allowed_per_pos[pos].size();
  if (numOfPositions == 0) return false;
  // if it's just MSD with one positional argument then size of the positions
  // is 2 (constraint.position + 1), minus one is 1, so it works for pass=1 but
  // for pass=2 it won't have any new positional decisions
  return pass < numOfPositions;
}

inline float pos_cost(task_data * data, ptag p) {
  if (p <= 0) return 0;
  return data->allowed_per_pos[p].size();
}

void run(Search::search& sch, vector<example*>& ec)
{ Search::predictor P(sch, (ptag)0);
  task_data * data = sch.get_task_data<task_data>();
  resize_tags(data, ec.size());
  size_t pass = 0, shift = 1;
  for (size_t i=0; i<ec.size(); i++)
  {
    float loss = 0;
    bool noloss = false;
    while (has_positional_decision(data, i, pass)) {
      ptag p = (ptag)shift;
      action oracle     = get_label(ec[i], pass);
      v_array<action> & allowed = get_allowed(data, pass, i);
      P.set_tag(p)
       .set_input(*ec[i])
       .set_oracle(oracle)
       .set_condition_range((ptag)(shift-1), sch.get_history_length(), 'p')
       .set_allowed(allowed);
      if (pass > 0 || i > 0) {
        P.add_condition(get_previous_ptag(data, i), 'a');
      }

      size_t prediction = P.predict();
      push_tag(data, i, p, prediction, pass);
      if (pass==0) {loss = prediction != oracle ?
                          pos_cost(data, prediction) : 0;
                    noloss = prediction != oracle ;}
      else if (!noloss)       {
        loss += (prediction != oracle);
      }
      // checks if it has a positional decisions for the next pass
      shift += 1;
      pass += 1;
    }
    pass = 0;
    sch.loss(loss);
    if (sch.output().good()){
      for(auto && a : data->tags[i]){
      sch.output() << sch.pretty_label((uint32_t)a.a)
                   << ' ';
      }
      sch.output() << '\n';
    }
  }
}
}



namespace SequenceSpanTask
{
enum EncodingType { BIO, BILOU };
// the format for the BIO encoding is:
//     label     description
//     1         "O" (out)
//     n even    begin X, where X is defined by n/2
//     n odd     in X, where X is (n-1)/2
//   thus, valid transitions are:
//     *       -> 1       (anything to OUT)
//     *       -> n even  (anything in BEGIN X)
//     n even  -> n+1     (BEGIN X to IN X)
//     n odd>1 -> n       (IN X to IN X)
// the format for the BILOU (begin, inside, last, out, unit-length) encoding is:
//     label     description
//     1         out
//     n>1: let m=n-2:
//       m % 4 == 0    unit-(m div 4)
//       m % 4 == 1    begin-(m div 4)
//       m % 4 == 2    in-(m div 4)
//       m % 4 == 3    last-(m div 4)
//   thus, valid transitions are:
//     1     -> 1; 2, 6, 10, ...; 3, 7, 11, ...         out to { out, unit-Y, begin-Y }       1
//     m%4=0 -> 1; 2, 6, 10, ..., 3, 7, 11, ...         unit-X to { out, unit-Y, begin-Y }    2, 6, 10, 14, ...
//     m%4=1 -> m+1, m+2                                begin-X to { in-X, last-X }           3, 7, 11, 15, ...
//     m%4=2 -> m, m+1                                  in-X to { in-X, last-X }              4, 8, 12, 16, ...
//     m%4=3 -> 1; 2, 6, 10, ...; 3, 7, 11, ...         last-X to { out, unit-Y, begin-Y }    5, 9, 13, 17, ...

inline action bilou_to_bio(action y)
{ return y / 2 + 1;  // out -> out, {unit,begin} -> begin; {in,last} -> in
}

void convert_bio_to_bilou(vector<example*> ec)
{ for (size_t n=0; n<ec.size(); n++)
  { MULTICLASS::label_t& ylab = ec[n]->l.multi;
    action y = ylab.label;
    action nexty = (n == ec.size()-1) ? 0 : ec[n+1]->l.multi.label;
    if (y == 1)   // do nothing
    {
    }
    else if (y % 2 == 0)     // this is a begin-X
    { if (nexty != y + 1) // should be unit
        ylab.label = (y/2 - 1) * 4 + 2;  // from 2 to 2, 4 to 6, 6 to 10, etc.
      else // should be begin-X
        ylab.label = (y/2 - 1) * 4 + 3;  // from 2 to 3, 4 to 7, 6 to 11, etc.
    }
    else if (y % 2 == 1)     // this is an in-X
    { if (nexty != y) // should be last
        ylab.label = (y-1) * 2 + 1;  // from 3 to 5, 5 to 9, 7 to 13, etc.
      else // should be in-X
        ylab.label = (y-1) * 2;      // from 3 to 4, 5 to 8, 7 to 12, etc.
    }
    assert( y == bilou_to_bio(ylab.label) );
  }
}

struct task_data
{ EncodingType encoding;
  v_array<action> allowed_actions;
  v_array<action> only_two_allowed;  // used for BILOU encoding
  size_t multipass;
};

void initialize(Search::search& sch, size_t& num_actions, po::variables_map& vm)
{ task_data * D = new task_data();
  po::options_description sspan_opts("search sequencespan options");
  sspan_opts.add_options()("search_span_bilou", "switch to (internal) BILOU encoding instead of BIO encoding");
  sspan_opts.add_options()("search_span_multipass", po::value<size_t>(&(D->multipass))->default_value(1), "do multiple passes");
  sch.add_program_options(vm, sspan_opts);

  if (vm.count("search_span_bilou"))
  { cerr << "switching to BILOU encoding for sequence span labeling" << endl;
    D->encoding = BILOU;
    num_actions = num_actions * 2 - 1;
  }
  else
    D->encoding = BIO;


  D->allowed_actions.erase();

  if (D->encoding == BIO)
  { D->allowed_actions.push_back(1);
    for (action l=2; l<num_actions; l+=2)
      D->allowed_actions.push_back(l);
    D->allowed_actions.push_back(1);  // push back an extra 1 that we can overwrite later if we want
  }
  else if (D->encoding == BILOU)
  { D->allowed_actions.push_back(1);
    for (action l=2; l<num_actions; l+=4)
    { D->allowed_actions.push_back(l);
      D->allowed_actions.push_back(l+1);
    }
    D->only_two_allowed.push_back(0);
    D->only_two_allowed.push_back(0);
  }

  sch.set_task_data<task_data>(D);
  sch.set_options( Search::AUTO_CONDITION_FEATURES  |    // automatically add history features to our examples, please
                   Search::AUTO_HAMMING_LOSS        |    // please just use hamming loss on individual predictions -- we won't declare loss
                   Search::EXAMPLES_DONT_CHANGE     |    // we don't do any internal example munging
                   0);
  sch.set_num_learners(D->multipass);
}

void finish(Search::search& sch)
{ task_data* D = sch.get_task_data<task_data>();
  D->allowed_actions.delete_v();
  D->only_two_allowed.delete_v();
  delete D;
}

void setup(Search::search& sch, vector<example*>& ec)
{ task_data& D = *sch.get_task_data<task_data>();
  if (D.encoding == BILOU)
    convert_bio_to_bilou(ec);
}

void takedown(Search::search& sch, vector<example*>& ec)
{ task_data& D = *sch.get_task_data<task_data>();

  if (D.encoding == BILOU)
    for (size_t n=0; n<ec.size(); n++)
    { MULTICLASS::label_t ylab = ec[n]->l.multi;
      ylab.label = bilou_to_bio(ylab.label);
    }
}

void run(Search::search& sch, vector<example*>& ec)
{ task_data& D = *sch.get_task_data<task_data>();
  v_array<action> * y_allowed = &(D.allowed_actions);
  Search::predictor P(sch, (ptag)0);
  for (size_t pass=1; pass<=D.multipass; pass++)
  { action last_prediction = 1;
    for (size_t i=0; i<ec.size(); i++)
    { action oracle = ec[i]->l.multi.label;
      size_t len = y_allowed->size();
      P.set_tag((ptag)i+1);
      P.set_learner_id(pass-1);
      if (D.encoding == BIO)
      { if      (last_prediction == 1)       P.set_allowed(y_allowed->begin(), len-1);
        else if (last_prediction % 2 == 0) { (*y_allowed)[len-1] = last_prediction+1; P.set_allowed(*y_allowed); }
        else                               { (*y_allowed)[len-1] = last_prediction;   P.set_allowed(*y_allowed); }
        if ((oracle > 1) && (oracle % 2 == 1) && (last_prediction != oracle) && (last_prediction != oracle-1))
          oracle = 1; // if we are supposed to I-X, but last wasn't B-X or I-X, then say O
      }
      else if (D.encoding == BILOU)
      { if ((last_prediction == 1) || ((last_prediction-2) % 4 == 0) || ((last_prediction-2) % 4 == 3))   // O or unit-X or last-X
        { P.set_allowed(D.allowed_actions);
          // we cannot allow in-X or last-X next
          if ((oracle > 1) && (((oracle-2) % 4 == 2) || ((oracle-2) % 4 == 3)))
            oracle = 1;
        }
        else     // begin-X or in-X
        { action other = ((last_prediction-2) % 4 == 1) ? (last_prediction+2) : last_prediction;
          P.set_allowed(last_prediction+1);
          P.add_allowed(other);
          if ((oracle != last_prediction+1) && (oracle != other))
            oracle = other;
        }
      }
      P.set_input(*ec[i]);
      P.set_condition_range((ptag)i, sch.get_history_length(), 'p');
      if (pass > 1) P.add_condition_range((ptag)(i+1+sch.get_history_length()), sch.get_history_length()+1, 'a');
      P.set_oracle(oracle);
      last_prediction = P.predict();

      if ((pass == D.multipass) && sch.output().good())
        sch.output() << ((D.encoding == BIO) ? last_prediction : bilou_to_bio(last_prediction)) << ' ';
    }
  }
}
}

namespace SequenceTaskCostToGo
{
void initialize(Search::search& sch, size_t& num_actions, po::variables_map& /*vm*/)
{ sch.set_options( Search::AUTO_CONDITION_FEATURES  |    // automatically add history features to our examples, please
                   Search::AUTO_HAMMING_LOSS        |    // please just use hamming loss on individual predictions -- we won't declare loss
                   Search::EXAMPLES_DONT_CHANGE     |    // we don't do any internal example munging
                   Search::ACTION_COSTS             |    // we'll provide cost-per-action (rather than oracle)
                   0);
  sch.set_task_data<size_t>(&num_actions);
}

void run(Search::search& sch, vector<example*>& ec)
{ size_t K = * sch.get_task_data<size_t>();
  float*costs = calloc_or_throw<float>(K);
  Search::predictor P(sch, (ptag)0);
  for (size_t i=0; i<ec.size(); i++)
  { action oracle     = ec[i]->l.multi.label;
    for (size_t k=0; k<K; k++) costs[k] = 1.;
    costs[oracle-1] = 0.;
    size_t prediction =
      P.set_tag((ptag)i+1)
      .set_input(*ec[i])
      .set_allowed(nullptr, costs, K)
      .set_condition_range((ptag)i, sch.get_history_length(), 'p')
      .predict();
    if (sch.output().good())
      sch.output() << sch.pretty_label((uint32_t)prediction) << ' ';
  }
  free(costs);
}
}

namespace ArgmaxTask
{
struct task_data
{ float false_negative_cost;
  float negative_weight;
  bool predict_max;
};

void initialize(Search::search& sch, size_t& /*num_actions*/, po::variables_map& vm)
{ task_data* D = new task_data();

  po::options_description argmax_opts("argmax options");
  argmax_opts.add_options()
  ("cost", po::value<float>(&(D->false_negative_cost))->default_value(10.0), "False Negative Cost")
  ("negative_weight", po::value<float>(&(D->negative_weight))->default_value(1), "Relative weight of negative examples")
  ("max", "Disable structure: just predict the max");
  sch.add_program_options(vm, argmax_opts);

  D->predict_max = vm.count("max") > 0;

  sch.set_task_data(D);

  if (D->predict_max)
    sch.set_options( Search::EXAMPLES_DONT_CHANGE );   // we don't do any internal example munging
  else
    sch.set_options( Search::AUTO_CONDITION_FEATURES |    // automatically add history features to our examples, please
                     Search::EXAMPLES_DONT_CHANGE );   // we don't do any internal example munging
}

void finish(Search::search& sch)
{ task_data* D = sch.get_task_data<task_data>();
  delete D;
}

void run(Search::search& sch, vector<example*>& ec)
{ task_data& D = *sch.get_task_data<task_data>();
  uint32_t max_prediction = 1;
  uint32_t max_label = 1;

  for(size_t i = 0; i < ec.size(); i++)
    max_label = max(ec[i]->l.multi.label, max_label);

  for (ptag i=0; i<ec.size(); i++)
  { // labels should be 1 or 2, and our output is MAX of all predicted values
    uint32_t oracle = D.predict_max ? max_label : ec[i]->l.multi.label;
    uint32_t prediction = sch.predict(*ec[i], i+1, &oracle, 1, &i, "p");

    max_prediction = max(prediction, max_prediction);
  }
  float loss = 0.;
  if (max_label > max_prediction)
    loss = D.false_negative_cost / D.negative_weight;
  else if (max_prediction > max_label)
    loss = 1.;
  sch.loss(loss);

  if (sch.output().good())
    sch.output() << max_prediction;
}
}


namespace SequenceTask_DemoLDF    // this is just to debug/show off how to do LDF
{
namespace CS=COST_SENSITIVE;
struct task_data
{ example* ldf_examples;
  size_t   num_actions;
};

void initialize(Search::search& sch, size_t& num_actions, po::variables_map& /*vm*/)
{ CS::wclass default_wclass = { 0., 0, 0., 0. };

  example* ldf_examples = VW::alloc_examples(sizeof(CS::label), num_actions);
  for (size_t a=0; a<num_actions; a++)
  { CS::label& lab = ldf_examples[a].l.cs;
    CS::cs_label.default_label(&lab);
    lab.costs.push_back(default_wclass);
  }

  task_data* data = &calloc_or_throw<task_data>();
  data->ldf_examples = ldf_examples;
  data->num_actions  = num_actions;

  sch.set_task_data<task_data>(data);
  sch.set_options( Search::AUTO_CONDITION_FEATURES |    // automatically add history features to our examples, please
                   Search::AUTO_HAMMING_LOSS       |    // please just use hamming loss on individual predictions -- we won't declare loss
                   Search::IS_LDF                  );   // we generate ldf examples
}

void finish(Search::search& sch)
{ task_data *data = sch.get_task_data<task_data>();
  for (size_t a=0; a<data->num_actions; a++)
    VW::dealloc_example(CS::cs_label.delete_label, data->ldf_examples[a]);
  free(data->ldf_examples);
  free(data);
}


// this is totally bogus for the example -- you'd never actually do this!
void my_update_example_indicies(Search::search& sch, bool audit, example* ec, uint64_t mult_amount, uint64_t plus_amount)
{ size_t ss = sch.get_stride_shift();
  for (features& fs : *ec)
    for (feature_index& idx : fs.indicies)
      idx = (((idx >> ss) * mult_amount) + plus_amount) << ss;
}

void run(Search::search& sch, vector<example*>& ec)
{ task_data *data = sch.get_task_data<task_data>();
  Search::predictor P(sch, (ptag)0);
  for (ptag i=0; i<ec.size(); i++)
  { for (size_t a=0; a<data->num_actions; a++)
    { if (sch.predictNeedsExample())   // we can skip this work if `predict` won't actually use the example data
      { VW::copy_example_data(false, &data->ldf_examples[a], ec[i]);  // copy but leave label alone!
        // now, offset it appropriately for the action id
        my_update_example_indicies(sch, true, &data->ldf_examples[a], 28904713, 4832917 * (uint64_t)a);
      }

      // regardless of whether the example is needed or not, the class info is needed
      CS::label& lab = data->ldf_examples[a].l.cs;
      // need to tell search what the action id is, so that it can add history features correctly!
      lab.costs[0].x = 0.;
      lab.costs[0].class_index = (uint64_t)a+1;
      lab.costs[0].partial_prediction = 0.;
      lab.costs[0].wap_value = 0.;
    }

    action oracle  = ec[i]->l.multi.label - 1;
    action pred_id = P.set_tag((ptag)(i+1)).set_input(data->ldf_examples, data->num_actions).set_oracle(oracle).set_condition_range(i, sch.get_history_length(), 'p').predict();
    action prediction = pred_id + 1;  // or ldf_examples[pred_id]->ld.costs[0].weight_index

    if (sch.output().good())
      sch.output() << prediction << ' ';
  }
}
}
