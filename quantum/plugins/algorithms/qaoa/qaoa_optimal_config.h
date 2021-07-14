#pragma once

#include <string>
#include "xacc_observable.hpp"
#include "IRProvider.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_observable.hpp"

class QAOAOptimalConfigEvaluator{

	public:
		static void evaluate(
				xacc::Observable* const m_costHamObs,
				std::shared_ptr<xacc::CompositeInstruction> evaled,
				xacc::Accelerator* const m_qpu,
				int nbSamples,
				int nbQubits,
				bool m_maximize,
				std::shared_ptr<xacc::AcceleratorBuffer> buffer,
				double* optimal_energy,
				std::string* opt_config,
				double* hit_rate);

	private:
		static double evaluate_assignment(xacc::Observable* const observable, std::string measurement);

};
