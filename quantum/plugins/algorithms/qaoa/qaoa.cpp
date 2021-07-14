/*******************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Thien Nguyen - initial API and implementation
 *   Milos Prokop - variable assignment mode
 *******************************************************************************/

#include "qaoa.hpp"
#include "qaoa_optimal_config.h"
#include <chrono> //delete

#include "xacc.hpp"
#include "Circuit.hpp"
#include "xacc_service.hpp"
#include "PauliOperator.hpp"
#include "xacc_observable.hpp"
#include "CompositeInstruction.hpp"
#include "AlgorithmGradientStrategy.hpp"

#include <cassert>
#include <iomanip>

namespace xacc {
namespace algorithm {

bool QAOA::initialize(const HeterogeneousMap &parameters) {
  bool initializeOk = true;
  // Hyper-parameters for QAOA:
  // (1) Accelerator (QPU)
  if (!parameters.pointerLikeExists<Accelerator>("accelerator")) {
    std::cout << "'accelerator' is required.\n";
    // We check all required params; hence don't early return on failure.
    initializeOk = false;
  }

  // (2) Classical optimizer
  if (!parameters.pointerLikeExists<Optimizer>("optimizer")) {
    std::cout << "'optimizer' is required.\n";
    initializeOk = false;
  }

  // (3) Number of mixing and cost function steps to use (default = 1)
  m_nbSteps = 1;
  if (parameters.keyExists<int>("steps")) {
    m_nbSteps = parameters.get<int>("steps");
  }

  // (4) Cost Hamiltonian to construct the max-cut cost Hamiltonian from.
  if (!parameters.pointerLikeExists<Observable>("observable")) {
        std::cout << "'observable' is required.\n";
        initializeOk = false;
    }

  // Default is Extended ParameterizedMode (less steps, more params)
  m_parameterizedMode = "Extended";
  if (parameters.stringExists("parameter-scheme")) {
    m_parameterizedMode = parameters.getString("parameter-scheme");
  }

  logStats = false;
  if (parameters.keyExists<std::function<void(int, double, double, double, std::string)>>("stats_func")) {
	  stats_func = parameters.get<std::function<void(int, double, double, double, std::string)>>("stats_func");
	  logStats = true;
  }

  // Determine the optimal variable assignment of the QUBO problem
  // If false, only optimal value is returned
  m_varAssignmentMode = false;
  if (parameters.keyExists<bool>("calc-var-assignment")) {
	  m_varAssignmentMode = parameters.get<bool>("calc-var-assignment");
  }

  m_simplifiedSimulationMode = false;
  if (parameters.keyExists<bool>("simplified-simulation")) {
	  m_simplifiedSimulationMode = parameters.get<bool>("simplified-simulation");
  }

  m_debugMsgs = false;
  if(parameters.keyExists<bool>("debugMsgs"))
	  m_debugMsgs = parameters.get<bool>("debugMsgs");

  if(m_varAssignmentMode){
    nbSamples = 1024;
    if(parameters.keyExists<int>("nbSamples"))
      nbSamples = parameters.get<int>("nbSamples");

    if(parameters.keyExists<int>("detailed_log_freq"))
      detailedLogFrequency = parameters.get<int>("detailed_log_freq");
  }

  if (initializeOk) {
    m_costHamObs = parameters.getPointerLike<Observable>("observable");
    m_qpu = parameters.getPointerLike<Accelerator>("accelerator");
    m_optimizer = parameters.getPointerLike<Optimizer>("optimizer");
    // Optional ref-hamiltonian
    m_refHamObs = nullptr;
    if (parameters.pointerLikeExists<Observable>("ref-ham")) {
      m_refHamObs = parameters.getPointerLike<Observable>("ref-ham");
    }
  }
     
  // Check if an initial composite instruction set has been provided
	if (parameters.pointerLikeExists<CompositeInstruction>("initial-state")) {
		  m_initial_state = std::shared_ptr<CompositeInstruction>(parameters.getPointerLike<CompositeInstruction>("initial-state"));
	}

  if(m_simplifiedSimulationMode){
      	m_costHamObs_pauli = dynamic_cast<quantum::PauliOperator*>(m_costHamObs);
      	if(!m_costHamObs_pauli){
      		std::cerr << "Hamiltonian could not be converted to Pauli operator. Using simplified-simulation=false";
      		m_simplifiedSimulationMode = false;
      	}

      	//initialize all H state explicitely. assumes QUEST is used
      	if (!parameters.pointerLikeExists<CompositeInstruction>("initial-state")) {
      		auto provider = getIRProvider("quantum");
			auto initial_program = provider->createComposite("qaoaInit");
			initial_program->addInstruction(provider->createInstruction("I", { 0 }));
      		m_initial_state = initial_program;
      	}

  }

  m_overlapTrick = false;
  zeroRefState = 0;
  if(parameters.keyExists<bool>("overlapTrick")){
	  m_overlapTrick = parameters.get<bool>("overlapTrick");
	  if(m_overlapTrick){
		  zeroRefState = parameters.get<int>("zeroRefState");
	  }
  }

  m_questHamExpectation = false;
    if(parameters.keyExists<bool>("questHamExpectation"))
    	m_questHamExpectation = parameters.get<bool>("questHamExpectation");
     

  // we need this for ADAPT-QAOA (Daniel)
  if (parameters.pointerLikeExists<CompositeInstruction>("ansatz")) {
    externalAnsatz =
        parameters.get<std::shared_ptr<CompositeInstruction>>("ansatz");
  }

  if (parameters.pointerLikeExists<AlgorithmGradientStrategy>(
          "gradient_strategy")) {
    gradientStrategy =
        parameters.get<std::shared_ptr<AlgorithmGradientStrategy>>(
            "gradient_strategy");
  }

  if (parameters.stringExists("gradient_strategy") &&
      !parameters.pointerLikeExists<AlgorithmGradientStrategy>(
          "gradient_strategy") &&
      m_optimizer->isGradientBased()) {
    gradientStrategy = xacc::getService<AlgorithmGradientStrategy>(
        parameters.getString("gradient_strategy"));
    gradientStrategy->initialize(parameters);
  }

  if ((parameters.stringExists("gradient_strategy") ||
       parameters.pointerLikeExists<AlgorithmGradientStrategy>(
           "gradient_strategy")) &&
      !m_optimizer->isGradientBased()) {
    xacc::warning(
        "Chosen optimizer does not support gradients. Using default.");
  }

  if (parameters.keyExists<bool>("maximize")) {
      m_maximize = parameters.get<bool>("maximize");
  }

  m_shuffleTerms = false;
  if (parameters.keyExists<bool>("shuffle-terms")) {
    m_shuffleTerms = parameters.get<bool>("shuffle-terms");
  }

  if (m_optimizer && m_optimizer->isGradientBased() &&
      gradientStrategy == nullptr) {
    // No gradient strategy was provided, just use autodiff.
    gradientStrategy = xacc::getService<AlgorithmGradientStrategy>("autodiff");
    gradientStrategy->initialize(
        {{"observable", xacc::as_shared_ptr(m_costHamObs)}});
  }
  return initializeOk;
}

//evaluate the energy of measurement
/*double QAOA::evaluate_assignment(xacc::Observable* const observable, std::string measurement) const{

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
}*/

const std::vector<std::string> QAOA::requiredParameters() const {
	return {"accelerator", "optimizer", "observable"};
}

void logd(std::string s){

	time_t now = time(0);
	struct tm  tstruct;
	char  buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
	std::cerr << "\033[0;36m" << "[[" << std::string(buf) << "]][DEBUG] " << s << "\033[0m\n";

}

void QAOA::execute(const std::shared_ptr<AcceleratorBuffer> buffer) const {
  const int nbQubits = buffer->size();
  // we need this for ADAPT-QAOA (Daniel)
  std::shared_ptr<CompositeInstruction> kernel;
  if (externalAnsatz) {
    kernel = externalAnsatz;
  } else {
      HeterogeneousMap m;
      kernel = std::dynamic_pointer_cast<CompositeInstruction>(
          xacc::getService<Instruction>("qaoa"));
      m.insert("nbQubits", nbQubits);
      m.insert("nbSteps", m_nbSteps);
      m.insert("ref-ham", m_refHamObs);
      m.insert("cost-ham", m_costHamObs);
      m.insert("parameter-scheme", m_parameterizedMode);
      if (m_initial_state){
          m.insert("initial-state", m_initial_state);
      }
      m.insert("shuffle-terms", m_shuffleTerms);
      kernel->expand(m);
  } 

  if(m_debugMsgs)
	  logd("Kernel loaded");

  
  // Observe the cost Hamiltonian:
  // OLD: 
  /*
  std::vector<std::shared_ptr<CompositeInstruction>> kernels;

  if(!m_simplifiedSimulationMode){

	  if(m_debugMsgs)
	  	  logd("Non-simplified kernel observation...");

	  kernels = m_costHamObs->observe(kernel);

	  if(m_debugMsgs)
	 	logd("Kernel observed");
  }
  else{

	  if(m_debugMsgs)
	  	  logd("Simplified kernel observation...");

	  auto gateRegistry = xacc::getService<IRProvider>("quantum");
	  auto gateFunction = gateRegistry->createComposite("I", kernel->getVariables());

	  gateFunction->setCoefficient(m_costHamObs->getIdentitySubTerm()->coefficient().real());

	  if (kernel->hasChildren()) {
		gateFunction->addInstruction(kernel->clone());
	  }

	  for (auto arg : kernel->getArguments()) {
		gateFunction->addArgument(arg, 0);
	  }

	  //std::vector<size_t> indices;
	  //for(size_t qbit_index=0; qbit_index < nbQubits; ++qbit_index){
	  //  indices.push_back(qbit_index);
	  //}
	  //gateFunction->addInstruction(gateRegistry->createInstruction("Measure", indices));

	  kernels.push_back(gateFunction);

	  if(m_questHamExpectation){

		  //do nothing

	  }else{ //append custom measures
		  std::vector<std::pair<std::pair<int, int>, double>> measurements_coeff;
		  for (std::map<std::string, quantum::Term>::iterator it = m_costHamObs_pauli->begin(); it != m_costHamObs_pauli->end(); ++it) {

				  quantum::Term spinInst = it->second;
				  auto termsMap = spinInst.ops();

				  int counter = 0, qbit_first;
				  for (auto &kv : termsMap) {

					  if (kv.second != "I" && !kv.second.empty()) {
						  if(kv.second != "Z"){
							  std::cerr << "Not implemented for hamiltonian composed of other than Z gates. Need to switch to simplified-simulation=false;\n";
							  throw;
					  }

					  if(counter++ == 0)
						  qbit_first = kv.first;
					  else{
						  measurements_coeff.push_back(std::pair(std::pair(qbit_first, kv.first), spinInst.coeff().real()));
						  counter = 0;
					  }

					}
				  }
				  if(counter == 1)
					  measurements_coeff.push_back(std::pair(std::pair(qbit_first, -1), spinInst.coeff().real())); //-1 means single qubit measurement
			  }

		  int i = 0;
		  for(auto &m : measurements_coeff){

			  auto measFunction = gateRegistry->createComposite("meas" + std::to_string(i), kernel->getVariables());

			  if(m.first.second < 0)
				  measFunction->addInstruction(gateRegistry->createInstruction("Measure", {static_cast<size_t>(m.first.first)}));
			  else
				  measFunction->addInstruction(gateRegistry->createInstruction("Measure", {static_cast<size_t>(m.first.first), static_cast<size_t>(m.first.second)}));

			  measFunction->setCoefficient(m.second);
			  kernels.push_back(measFunction);

		  }
	  }

	  if(m_debugMsgs)
	  	logd("Kernel observed");
  }*/
//NEW
  // Handle Max-cut optimization on shots-based backends (including physical
  // backends). We only want to execute a single circuit for observable with all
  // commuting terms such as the maxcut Hamiltonian.
  // Limitation: this grouping cannot handle gradient strategy at the moment.
  // Observe the cost Hamiltonian:
  auto kernels = [&] {
    if (dynamic_cast<xacc::quantum::PauliOperator *>(m_costHamObs)) {
      return m_costHamObs->observe(kernel, {{"accelerator", m_qpu}});
    } else {
      return m_costHamObs->observe(kernel);
    }
  }();

  // Grouping is possible (no gradient strategy)
  // TODO: Gradient strategy to handle grouping as well.
  int iterCount = 0;
//NEW CODE
  if (m_costHamObs->getNonIdentitySubTerms().size() > 1 &&
      kernels.size() == 1 && !gradientStrategy) {
    OptFunction f(
        [&, this](const std::vector<double> &x, std::vector<double> &dx) {
          auto tmpBuffer = xacc::qalloc(buffer->size());
          std::vector<std::shared_ptr<CompositeInstruction>> fsToExec{
              kernels[0]->operator()(x)};
          m_qpu->execute(tmpBuffer, fsToExec);
          double energy = m_costHamObs->postProcess(tmpBuffer);
          // We will only have one child buffer for each parameter set.
          assert(tmpBuffer->getChildren().size() == 1);
          auto result_buf = tmpBuffer->getChildren()[0];
          result_buf->addExtraInfo("parameters", x);
          result_buf->addExtraInfo("energy", energy);
          buffer->appendChild("Iter" + std::to_string(iterCount), result_buf);

          std::stringstream ss;

          ss << "Iter " << iterCount << ": E("
             << (!x.empty() ? std::to_string(x[0]) : "");
          for (int i = 1; i < x.size(); i++) {
            ss << "," << std::setprecision(3) << x[i];
            if (i > 4) {
              // Don't print too many params
              ss << ", ...";
              break;
            }
          }
          ss << ") = " << std::setprecision(12) << energy;
          xacc::info(ss.str());
          iterCount++;
          if (m_maximize)
            energy *= -1.0;
          return energy;
        }, kernel->nVariables());
    auto result = m_optimizer->optimize(f);
    // Reports the final cost:
    double finalCost = result.first;
    if (m_maximize)
      finalCost *= -1.0;
    buffer->addExtraInfo("opt-val", ExtraInfo(finalCost));
    buffer->addExtraInfo("opt-params", ExtraInfo(result.second));
    return;
  }
//END OF NEW CODE
  // Construct the optimizer/minimizer:
  OptFunction f(
      [&, this](const std::vector<double> &x, std::vector<double> &dx) {
        std::vector<double> coefficients;
        std::vector<std::string> kernelNames;
        std::vector<std::shared_ptr<CompositeInstruction>> fsToExec;

        double identityCoeff = 0.0;
        int nInstructionsEnergy = 0, nInstructionsGradient = 0;
        for (auto &f : kernels) {
          kernelNames.push_back(f->name());
          std::complex<double> coeff = f->getCoefficient();

          int nFunctionInstructions = 0;
          if (f->getInstruction(0)->isComposite()) {
            nFunctionInstructions =
                kernel->nInstructions() + f->nInstructions() - 1;
          } else {
            nFunctionInstructions = f->nInstructions();
          }

          if (nFunctionInstructions > kernel->nInstructions()) {
            auto evaled = f->operator()(x);
            fsToExec.push_back(evaled);
            coefficients.push_back(std::real(coeff));
          } else {
            identityCoeff += std::real(coeff);
          }
        }

        // enables gradients (Daniel)
        if (gradientStrategy) {

          auto gradFsToExec =
              gradientStrategy->getGradientExecutions(kernel, x);
          // Add gradient instructions to be sent to the qpu
          nInstructionsEnergy = fsToExec.size();
          nInstructionsGradient = gradFsToExec.size();
          for (auto inst : gradFsToExec) {
            fsToExec.push_back(inst);
          }
          xacc::info("Number of instructions for energy calculation: " +
                     std::to_string(nInstructionsEnergy));
          xacc::info("Number of instructions for gradient calculation: " +
                     std::to_string(nInstructionsGradient));
        }

        auto tmpBuffer = xacc::qalloc(buffer->size());
        m_qpu->execute(tmpBuffer, fsToExec);
        auto buffers = tmpBuffer->getChildren();

        double energy = identityCoeff;
        auto idBuffer = xacc::qalloc(buffer->size());
        idBuffer->addExtraInfo("coefficient", identityCoeff);
        idBuffer->setName("I");
        idBuffer->addExtraInfo("kernel", "I");
        idBuffer->addExtraInfo("parameters", x);
        idBuffer->addExtraInfo("exp-val-z", 1.0);
        buffer->appendChild("I", idBuffer);

        if (gradientStrategy) { // gradient-based optimization

          for (int i = 0; i < nInstructionsEnergy; i++) { // compute energy
            auto expval = buffers[i]->getExpectationValueZ();
            energy += expval * coefficients[i];
            buffers[i]->addExtraInfo("coefficient", coefficients[i]);
            buffers[i]->addExtraInfo("kernel", fsToExec[i]->name());
            buffers[i]->addExtraInfo("exp-val-z", expval);
            buffers[i]->addExtraInfo("parameters", x);
            buffer->appendChild(fsToExec[i]->name(), buffers[i]);
          }

          std::stringstream ss;
          ss << std::setprecision(12) << "Current Energy: " << energy;
          xacc::info(ss.str());
          ss.str(std::string());

          // If gradientStrategy is numerical, pass the energy
          // We subtract the identityCoeff from the energy
          // instead of passing the energy because the gradients
          // only take the coefficients of parameterized instructions
          if (gradientStrategy->isNumerical()) {
            gradientStrategy->setFunctionValue(energy - identityCoeff);
          }

          // update gradient vector
          gradientStrategy->compute(
              dx, std::vector<std::shared_ptr<AcceleratorBuffer>>(
                      buffers.begin() + nInstructionsEnergy, buffers.end()));

        } else { // normal QAOA run

          for (int i = 0; i < buffers.size(); i++) {
            auto expval = buffers[i]->getExpectationValueZ();
            energy += expval * coefficients[i];
            buffers[i]->addExtraInfo("coefficient", coefficients[i]);
            buffers[i]->addExtraInfo("kernel", fsToExec[i]->name());
            buffers[i]->addExtraInfo("exp-val-z", expval);
            buffers[i]->addExtraInfo("parameters", x);
            buffer->appendChild(fsToExec[i]->name(), buffers[i]);
          }
        }
   
        std::stringstream ss;
        iterCount++;
        ss << "Iter " << iterCount << ": E("
           << (!x.empty() ? std::to_string(x[0]) : "");
        for (int i = 1; i < x.size(); i++) {
          ss << "," << std::setprecision(3) << x[i];
          if (i > 4) {
            // Don't print too many params
            ss << ", ...";
            break;
          }
        }
        ss << ") = " << std::setprecision(12) << energy;
        xacc::info(ss.str());
        
      if (m_maximize) energy *= -1.0;

      if(logStats)
      	stats_func(4, energy, 0, 0, "");

      return energy;
      },
      kernel->nVariables());
//NEW CODE
  int iterationCounter = 1;
  OptFunction f_simplified_simulation_mode(
        [&, this](const std::vector<double> &x, std::vector<double> &dx) {

		  if(m_debugMsgs)
			 logd("Simplified_mode iteration");

		  if(logStats)
			stats_func(3, 0, 0, 0, ""); //optimizer finish

          std::vector<double> coefficients;
          std::vector<std::string> kernelNames;
          std::vector<std::shared_ptr<CompositeInstruction>> fsToExec;

          double identityCoeff = m_costHamObs -> getIdentitySubTerm() -> coefficient().real();
          //int nInstructionsEnergy = 0, nInstructionsGradient = 0;

          if(m_debugMsgs)
          	logd("Substituting parameters...");
          for (auto &f : kernels) {

            kernelNames.push_back(f->name());
            std::complex<double> coeff = f->getCoefficient();

            //if (f->name() != "I") {
              auto provider = getIRProvider("quantum");
              auto evaled = f->operator()(x);
              fsToExec.push_back(evaled);
              if(m_overlapTrick)
            	  evaled->addInstructions({provider->createInstruction("OverlapInstruction", {0}, {(double)zeroRefState})});

              coefficients.push_back(std::real(coeff));

            //}

/*              if(m_overlapTrick){
            	  auto provider = getIRProvider("quantum");
            	  auto overlap_calc = provider->createComposite("overlap_calc", {});
            	  overlap_calc->addInstructions({provider->createInstruction("OverlapInstruction", {}, {}, {{"zeroRefState", zeroRefState}})});
            	  fsToExec.push_back(overlap_calc);
              }*/

          }

          if(m_debugMsgs)
         	 logd("Parameters substituted.");

          // enables gradients (Daniel)
          if (gradientStrategy) {

            /*auto gradFsToExec =
                gradientStrategy->getGradientExecutions(kernel, x);
            // Add gradient instructions to be sent to the qpu
            nInstructionsEnergy = fsToExec.size();
            nInstructionsGradient = gradFsToExec.size();
            for (auto inst : gradFsToExec) {
              fsToExec.push_back(inst);
            }
            xacc::info("Number of instructions for energy calculation: " +
                       std::to_string(nInstructionsEnergy));
            xacc::info("Number of instructions for gradient calculation: " +
                       std::to_string(nInstructionsGradient));*/
          }

          /**for(auto f:fsToExec)
          	std::cout << f->toString() << "\n";
          throw;*/

          //xacc::setOption("quest-verbose", "true");

          if(m_debugMsgs)
          	logd("Allocating temp space for " + std::to_string(buffer->size()) + " qubits");

          auto tmpBuffer = xacc::qalloc(buffer->size());

          const std::vector<double>* coeff_ptr = static_cast<const std::vector<double>*>(&coefficients);
		  std::stringstream ss_coeff_ptr;
		  ss_coeff_ptr << coeff_ptr;
		  xacc::setOption("coeff_ptr", ss_coeff_ptr.str());

		  if(m_debugMsgs)
		     logd("Coeff_ptr set");

		  if(logStats)
			  stats_func(0, 0, 0, 0, ""); //startQuantumIterLog

		  logd("Starting execution on m_qpu");

		  auto then = std::chrono::system_clock::now();
          m_qpu->execute(tmpBuffer, fsToExec);
          auto now = std::chrono::system_clock::now();
          logd("Finished execution on m_qpu");
          std::stringstream ss;
          ss << "Took " << std::chrono::duration_cast<std::chrono::duration<float>>(now-then).count() << "s and calculated: " << (*tmpBuffer)["exp-val-z"].as<double>();
          logd(ss.str());
          //throw;

          if(logStats)
          	  stats_func(1, 0, 0, 0, ""); //finishQuantumIterLog

                  //auto buffers = tmpBuffer->getChildren();

          double energy;
		  if (gradientStrategy) { // gradient-based optimization

			/*energy = (*buffer)["exp-val-z"].as<double>();

			std::stringstream ss;
			ss << std::setprecision(12) << "Current Energy: " << energy;
			xacc::info(ss.str());
			ss.str(std::string());

			// If gradientStrategy is numerical, pass the energy
			// We subtract the identityCoeff from the energy
			// instead of passing the energy because the gradients
			// only take the coefficients of parameterized instructions
			if (gradientStrategy->isNumerical()) {
				gradientStrategy->setFunctionValue(energy - identityCoeff);
				}

				// update gradient vector
				gradientStrategy->compute(dx, std::vector<std::shared_ptr<AcceleratorBuffer>>(
						  buffers.begin() + nInstructionsEnergy, buffers.end()));*/

			}else{ // normal QAOA run
				energy = (*tmpBuffer)["exp-val-z"].as<double>();
            }

         /* std::stringstream ss;
          iterCount++;
          ss << "Iter " << iterCount << ": E("
             << (!x.empty() ? std::to_string(x[0]) : "");
          for (int i = 1; i < x.size(); i++) {
            ss << "," << std::setprecision(3) << x[i];
            if (i > 4) {
              // Don't print too many params
              ss << ", ...";
              break;
            }
          }
          ss << ") = " << std::setprecision(12) << energy;
          xacc::info(ss.str());*/

          if (m_maximize) energy *= -1.0;

          if (detailedLogFrequency > 0 && iterationCounter % detailedLogFrequency == 0){ //detailed log

        	  logd("Detailed log");

        	  double optimal_energy, hit_rate;
        	  std::string opt_config;

			  QAOAOptimalConfigEvaluator::evaluate(
					  m_costHamObs,
					  kernel->operator()(x),
					  m_qpu,
					  nbSamples,
					  nbQubits,
					  m_maximize,
					  buffer,
					  &optimal_energy,
					  &opt_config,
					  &hit_rate);

			  stats_func(4, energy, optimal_energy, hit_rate, opt_config);

			  if(logStats)
				stats_func(2, 0, 0, 0, ""); //optimizer start

          }else if(logStats){
        	stats_func(2, 0, 0, 0, ""); //optimizer start
          	stats_func(5, energy, 0, 0, ""); //final report
          }

          iterationCounter++;
          logd("Simplified iteration finish");
          return energy;
        },
        kernel->nVariables());

  if(m_debugMsgs)
 	logd("Starting optimization..");

  OptResult result;
  if(m_simplifiedSimulationMode){
	  try{
		  result = m_optimizer->optimize(f_simplified_simulation_mode);
	  }catch(...){
		  logd("FASUL");
	  }
  }

  else
	  result = m_optimizer->optimize(f);
//END OF NEW CODE
  
  // Reports the final cost:
  double finalCost = result.first;
  if (m_maximize) finalCost *= -1.0;

  if(m_varAssignmentMode){

	  double optimal_energy, hit_rate;
	  std::string opt_config;

	  QAOAOptimalConfigEvaluator::evaluate(
			  m_costHamObs,
			  kernel->operator()(result.second),
			  m_qpu,
			  nbSamples,
			  nbQubits,
			  m_maximize,
			  buffer,
			  &optimal_energy,
			  &opt_config,
			  &hit_rate);

	  if( detailedLogFrequency > 0){
		  stats_func(4, finalCost, optimal_energy, hit_rate, opt_config);
	  }

	  std::cout << "Final opt-val: " << optimal_energy << "\n";
	  std::cout << "Final opt-config: " << opt_config << "\n";
	  std::cout << "Final hit-rate: " << hit_rate << "\n";


	  buffer->addExtraInfo("expected-val", ExtraInfo(finalCost));
	  buffer->addExtraInfo("opt-val", ExtraInfo(optimal_energy));
	  buffer->addExtraInfo("opt-config", opt_config);
	  buffer->addExtraInfo("hit_rate", ExtraInfo(hit_rate));
	  buffer->addExtraInfo("opt-params", ExtraInfo(result.second));

  }else{
	  buffer->addExtraInfo("opt-val", ExtraInfo(finalCost));
	  buffer->addExtraInfo("opt-params", ExtraInfo(result.second));
  }

}

std::vector<double>
QAOA::execute(const std::shared_ptr<AcceleratorBuffer> buffer,
              const std::vector<double> &x) {
  const int nbQubits = buffer->size();
  std::shared_ptr<CompositeInstruction> kernel;
  if (externalAnsatz) {
    kernel = externalAnsatz;
  } else if (m_single_exec_kernel) {
    kernel = m_single_exec_kernel;
  } else {
    HeterogeneousMap m;
    kernel = std::dynamic_pointer_cast<CompositeInstruction>(
          xacc::getService<Instruction>("qaoa"));
    m.insert("nbQubits", nbQubits);
    m.insert("nbSteps", m_nbSteps);
    m.insert("ref-ham", m_refHamObs);
    m.insert("cost-ham", m_costHamObs);
    m.insert("parameter-scheme", m_parameterizedMode);
    if (m_initial_state){
        m.insert("initial-state", m_initial_state);
    }
    m.insert("shuffle-terms", m_shuffleTerms);
    kernel->expand(m);
    // save this kernel for future calls to execute
    m_single_exec_kernel = kernel;
  }

  // Observe the cost Hamiltonian:
  //OLD auto kernels = m_costHamObs->observe(kernel);
  //NEW
  // Observe the cost Hamiltonian, with the input Accelerator:
  // i.e. perform grouping (e.g. max-cut QAOA, Pauli) if possible:
  auto kernels = [&] {
    if (dynamic_cast<xacc::quantum::PauliOperator *>(m_costHamObs)) {
      return m_costHamObs->observe(kernel, {{"accelerator", m_qpu}});
    } else {
      return m_costHamObs->observe(kernel);
    }
  }();

  if (m_costHamObs->getNonIdentitySubTerms().size() > 1 &&
      kernels.size() == 1) {
    // Grouping was done:
    // just execute the single observed kernel:
    std::vector<std::shared_ptr<CompositeInstruction>> fsToExec{
        kernels[0]->operator()(x)};
    m_qpu->execute(buffer, fsToExec);
    const double finalCost = m_costHamObs->postProcess(buffer);
    // std::cout << "Compute energy from grouping: " << finalCost << "\n";
    return { finalCost };
  }
  //END OF NEW
  std::vector<double> coefficients;
  std::vector<std::string> kernelNames;
  std::vector<std::shared_ptr<CompositeInstruction>> fsToExec;

  double identityCoeff = 0.0;
  for (auto &f : kernels) {
    kernelNames.push_back(f->name());
    std::complex<double> coeff = f->getCoefficient();

    int nFunctionInstructions = 0;
    if (f->getInstruction(0)->isComposite()) {
      nFunctionInstructions = kernel->nInstructions() + f->nInstructions() - 1;
    } else {
      nFunctionInstructions = f->nInstructions();
    }

    if (nFunctionInstructions > kernel->nInstructions()) {
      auto evaled = f->operator()(x);
      fsToExec.push_back(evaled);
      coefficients.push_back(std::real(coeff));
    } else {
      identityCoeff += std::real(coeff);
    }
  }

  auto tmpBuffer = xacc::qalloc(buffer->size());
  m_qpu->execute(tmpBuffer, fsToExec);
  auto buffers = tmpBuffer->getChildren();

  double energy = identityCoeff;
  auto idBuffer = xacc::qalloc(buffer->size());
  idBuffer->addExtraInfo("coefficient", identityCoeff);
  idBuffer->setName("I");
  idBuffer->addExtraInfo("kernel", "I");
  idBuffer->addExtraInfo("parameters", x);
  idBuffer->addExtraInfo("exp-val-z", 1.0);
  buffer->appendChild("I", idBuffer);

  for (int i = 0; i < buffers.size(); i++) {
    auto expval = buffers[i]->getExpectationValueZ();
    energy += expval * coefficients[i];
    buffers[i]->addExtraInfo("coefficient", coefficients[i]);
    buffers[i]->addExtraInfo("kernel", fsToExec[i]->name());
    buffers[i]->addExtraInfo("exp-val-z", expval);
    buffers[i]->addExtraInfo("parameters", x);
    buffer->appendChild(fsToExec[i]->name(), buffers[i]);
  }
  
  // WARNING: Removing the parameter shifting here. Remember for later
  // in case of any tests that fail. 
  const double finalCost = energy;
    //   m_maxcutProblem ? (-0.5 * energy +
    //                      0.5 * (m_costHamObs->getNonIdentitySubTerms().size()))
    //                   : energy;
  return {finalCost};
}

} // namespace algorithm
} // namespace xacc
