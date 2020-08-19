import preprocessing as pp
import predictor as pr
import question_generator as qg
import distractor_generator as dg
import sentence_ranker as sr

import random

models = ['gaussian_naive_bayes', 'logistic_regression']

# Generate Questions
def generateQuestions(text, count):
	# Extract words
	text = sr.rearrangeByRank(text)
	df = pp.generateDf(text)
	wordsDf = pp.prepareDf(df)

	for model in models:
		print("\n#####################################\n")
		print(model)
		print("\n#####################################\n")

		# Predict
		labeledAnswers = pr.predictWords(wordsDf, df, model)

		# Transform questions
		qaPairs = qg.addQuestions(labeledAnswers, text)

		# Pick the best questions
		orderedQaPairs = qg.sortAnswers(qaPairs)

		# Generate distractors
		questions = dg.addDistractors(orderedQaPairs[:count], 3)

		# Print
		print('Text:')
		print(text + '\n')
		for index, question in enumerate(questions):
			print('Question ' + str(index + 1) + ':')
			print(question['question'])

			print('Wh Question ' + str(index + 1) + ':')
			print(question['whQuestion'])

			options = []
			options.append(question['answer'])
			options.extend(question['distractors'])
			random.shuffle(options)

			print()
			for option in options:
				print(option)
			print()

text = "Summary Changes of state are examples of phase changes, or phase transitions. All phase changes are accompanied by changes in the energy of a system. Changes from a more-ordered state to a less-ordered state (such as a liquid to a gas) areendothermic. Changes from a less-ordered state to a more-ordered state (such as a liquid to a solid) are always exothermic. The conversion of a solid to a liquid is called fusion (or melting). The energy required to melt 1 mol of a substance is its enthalpy of fusion (\u0394Hfus). The energy change required to vaporize 1 mol of a substance is the enthalpy of vaporization (\u0394Hvap). The direct conversion of a solid to a gas is sublimation. The amount of energy needed to sublime 1 mol of a substance is its enthalpy of sublimation (\u0394Hsub) and is the sum of the enthalpies of fusion and vaporization. Plots of the temperature of a substance versus heat added or versus heating time at a constant rate of heating are calledheating curves. Heating curves relate temperature changes to phase transitions. A superheated liquid, a liquid at a temperature and pressure at which it should be a gas, is not stable. A cooling curve is not exactly the reverse of the heating curve because many liquids do not freeze at the expected temperature. Instead, they form a supercooled liquid, a metastable liquid phase that exists below the normal melting point. Supercooled liquids usually crystallize on standing, or adding a seed crystal of the same or another substance can induce crystallization."

generateQuestions(text, 10)

text2 = "One way to keep iron from corroding is to keep it painted. The layer of paint prevents the water and oxygen necessary for rust formation from coming into contact with the iron. As long as the paint remains intact, the iron is protected from corrosion. Other strategies include alloying the iron with other metals. For example, stainless steel is mostly iron with a bit of chromium. The chromium tends to collect near the surface, where it forms an oxide layer that protects the iron. Zinc-plated or galvanized iron uses a different strategy. Zinc is more easily oxidized than iron because zinc has a lower reduction potential. Since zinc has a lower reduction potential, it is a more active metal. Thus, even if the zinc coating is scratched, the zinc will still oxidize before the iron. This suggests that this approach should work with other active metals. Another important way to protect metal is to make it the cathode in a galvanic cell. This is cathodic protection and can be used for metals other than just iron. For example, the rusting of underground iron storage tanks and pipes can be prevented or greatly reduced by connecting them to a more active metal such as zinc or magnesium (Figure 17.18). This is also used to protect the metal parts in water heaters. The more active metals (lower reduction potential) are called sacrificial anodes because as they get used up as they corrode (oxidize) at the anode. The metal being protected serves as the cathode, and so does not oxidize (corrode). When the anodes are properly monitored and periodically replaced, the useful lifetime of the iron storage tank can be greatly extended."

generateQuestions(text2, 10)

text3 = "Ganglia A ganglion is a group of neuron cell bodies in the periphery. Ganglia can be categorized, for the most part, as either sensory ganglia or autonomic ganglia, referring to their primary functions. The most common type of sensory ganglion is a dorsal (posterior) root ganglion. These ganglia are the cell bodies of neurons with axons that are sensory endings in the periphery, such as in the skin, and that extend into the CNS through the dorsal nerve root. The ganglion is an enlargement of the nerve root. Under microscopic inspection, it can be seen to include the cell bodies of the neurons, as well as bundles of fibers that are the posterior nerve root (Figure 13.19). The cells of the dorsal root ganglion are unipolar cells, classifying them by shape. Also, the small round nuclei of satellite cells can be seen surrounding\u2014as if they were orbiting\u2014the neuron cell bodies."

generateQuestions(text3, 10)
