#include "qaoa_optimal_config.h"

void QAOAOptimalConfigEvaluator::evaluate(
		xacc::Observable* const m_costHamObs,
		std::shared_ptr<xacc::CompositeInstruction> evaled,
		xacc::Accelerator* const m_qpu,
		int nbSamples,
		int nbQubits,
		bool m_maximize,
		std::shared_ptr<xacc::AcceleratorBuffer> buffer,
		double* optimal_energy,
		std::string* opt_config,
		double* hit_rate){

	/*typedef std::pair<int, double> meas_freq_eval;
	typedef std::pair<std::string, meas_freq_eval> measurement;

	auto provider = xacc::getIRProvider("quantum");
	//auto evaled = kernel->operator()(result.second);

	int tempShotsNumber = m_qpu->getShotsNumber();
	m_qpu->updateShotsNumber(nbSamples);

	std::vector<size_t> indices;
	for(size_t i=0; i < nbQubits; ++i){
	  indices.push_back(i);
	}

	auto meas = provider->createInstruction("Measure", indices);
	evaled->addInstructions({meas});

	m_qpu->execute(buffer, evaled);
	std::vector<measurement> measurements;

	for(std::pair<std::string, int> meas : buffer->getMeasurementCounts()){

	  bool found = false;
	  for(auto &instance : measurements)
		  if(instance.first == meas.first){
			  found = true;
			  break;
		  }

	  if(!found){
		  measurements.push_back(measurement(meas.first, // bit string
								  meas_freq_eval(meas.second, //frequency
										  evaluate_assignment(m_costHamObs, meas.first))));
	  }
	}

	if(m_maximize)
	  std::sort(measurements.begin(), measurements.end(),
		  [](const std::pair<std::string, std::pair<int, double>>& a, const std::pair<std::string, std::pair<int,int>>& b) {
			  //sort by global value
			  return a.second.second > b.second.second;
	  });

	else
	  std::sort(measurements.begin(), measurements.end(),
		  [](const std::pair<std::string, std::pair<int, double>>& a, const std::pair<std::string, std::pair<int,int>>& b) {
			  //sort by global value
			  return a.second.second < b.second.second;
	});

	int i = 0;
	int hits = 0;
	while(i < measurements.size() && abs(measurements[i++].second.second - measurements[0].second.second)<10e-1){
	  hits += measurements[i-1].second.first;
	}

	*optimal_energy = measurements[0].second.second;
	*opt_config = measurements[0].first;
	*hit_rate = hits / double(nbSamples);

	m_qpu->updateShotsNumber(tempShotsNumber);*/

	std::cerr<<"Calculating opt assignment\n";

	std::vector<std::shared_ptr<xacc::CompositeInstruction>> functions;
	//functions.push_back(evaled);
	buffer->addExtraInfo("finalConfigEvaluator", true);
	m_qpu->execute(buffer, functions);
	*opt_config = (*buffer)["opt_config_found"].as<std::string>();
	*optimal_energy = (*buffer)["opt_config_energy"].as<double>();
	*hit_rate = (*buffer)["opt_hit_rate"].as<double>();;


	//*optimal_energy = 0;
	//*opt_config = "00000";
	//*hit_rate = 0;

}

double QAOAOptimalConfigEvaluator::evaluate_assignment(xacc::Observable* const observable, std::string measurement){

	double result =	observable->getIdentitySubTerm()->coefficient().real();
	for(auto &term: observable->getNonIdentitySubTerms()){

		//get the real part, imag expected to be 0
		double coeff = term->coefficient().real();
		//parse to get hamiltonian terms
		std::string term_str = term->toString();
		std::vector<int> qubit_indices;

		char z_coeff[6]; // Number of ciphers in qubit index are expected to fit here.
		int z_index = 0;

		int term_i = 0;

		for(size_t i = term_str.length()-1; i > 0; --i){

			char c = term_str[i];
			if(isdigit(c))
				z_coeff[5-z_index++] = c;
			else if(c == 'Z'){
				int val = 0;
				for(; z_index>0; --z_index){
					val += pow(10, z_index-1) * (z_coeff[5-z_index+1] - '0');
				}

				qubit_indices.push_back(val);
				z_index = 0;

			}
			else if(c == ' '){
				if(++term_i == 2)
					break;
			}
			else{
				break;
			}
		}
		int multiplier = 1;

		int num_qubits=measurement.size();
		 //CRITICAL PART; STRING QUBIT ORDERING IS REVERSED!!
		for(auto &qubit_index: qubit_indices)
			multiplier *= (measurement[num_qubits-1-qubit_index] == '0') ? 1 : -1;

		result += multiplier * coeff;
	}

	return result;

}
