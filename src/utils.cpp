#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <thread>

// #include <SFML/Graphics.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace _random{

	double random(){
		return rand()%10000/10000.0;
	}

	void random_checker(){
		double top = 0.0;
		double bottom = 0.0;

		while (true){
			double x = random();
			double y = random();

			if (x*x + y*y <= 1){
				top = top + 1.0;
			}bottom = bottom + 1.0;


			double pi = 4 * top/bottom;
			double err = (pi - M_PI) * (pi - M_PI);

			double score = 1 - err;
			std::cout << "score = " << score*100 << "%" << std::endl;
		}
	}

	double normal0_1(){
		double U = random();
		double V = random();
		return sqrt(-2.0 * log(U)) * cos(2.0 * M_PI * V);
	}

	void normal_checker(){

		double top = 0.0;
		double bottom = 0.0;

		while (true){
			double x = normal0_1();
			double y = normal0_1();

			if (x*x + y*y <= 1){
				top = top + 1.0;
			}bottom = bottom + 1.0;


			double ratio = top/bottom;
			double err = (ratio - 0.67) * (ratio - 0.67);
			double score = 1 - err * err;
			std::cout << "score = " << score*100 << "%" << std::endl;
		}
	}

	double normal(double mean, double std){
		double z = normal0_1();
		return mean + std * z;
	}
}

namespace feedforward{

	class synapse{
	public:

        std::tuple<int, int> start_loc;
        std::tuple<int, int> end_loc;        

        double weight;
        bool enabled;
    
        synapse(){
            this->weight = _random::normal(0, 1);
            this->enabled = true;
        }

        synapse(bool enabled){
            this->weight = _random::normal(0, 1);
            this->enabled = enabled;
        }

        void mutate(double std){
            this->weight = this->weight + _random::normal(0, std);
        }   

        synapse clone(){
            synapse s(this->enabled);
            s.weight = this->weight;
			s.start_loc = this->start_loc;
			s.end_loc = this->end_loc;
            return s;
        }

    };

	class neuron{
	public:

        double bias;
        double holder;
        bool from_origin;

		std::tuple<int, int> location;

        neuron(){
            this->bias = _random::normal(0, 1);
            this->holder = 0;
            this->from_origin = true;
        }

        neuron(double holder){
            this->bias = _random::normal(0, 1);
            this->holder = holder;
            this->from_origin = true;
        }

        double activation(double z){
            return tanh(z);
        }

        void forward(){
            this->holder = this->activation(this->holder + this->bias);
        }

        void mutate(double std){
            this->bias = this->bias + _random::normal(0, std);
        }

        neuron clone(){
            neuron n;
            n.holder = 0;
            n.bias = this->bias;
			n.location = this->location;
            return n;
        }
	};

    class network{
    public:

        // brain stores the vector of layers
        // layer is a vector of neurons present in a single level
        /* 
            [
                [layer 0],
                [layer 1],
                    .
                    .
                    .
                [layer n],
            ]
        */
        std::vector<std::vector<neuron>> brain; 

        // nerves are the synapses from each layer in forward direction
        /* 
            [
                [from layer 0],
                [from layer 1],
                [from layer 2],
                      .
                      .
                      .
                [from layer n-1]
            ]
        */
        std::vector<std::vector<synapse>> nerves;

		network(){

		}

        network(std::vector<int> layer_structure, double density_coefficient){

            // initialising the brain, layer wise rep.
            for (int l = 0; l < layer_structure.size(); ++l){
                std::vector<neuron> layer;
                for (int r = 0; r < layer_structure[l]; ++r){
                    neuron cell(0);
					cell.location = std::make_tuple(l, r);
                    layer.push_back(cell);
                }this->brain.push_back(layer);
            }

            // initialising the nervous system, layer wise rep.
            for (int src_layer = 0; src_layer < layer_structure.size()-1; ++src_layer){
                std::vector<synapse> from_layer;
                for (int dst_layer = src_layer+1; dst_layer < layer_structure.size(); ++dst_layer){
                    for (int s = 0; s < layer_structure[src_layer]; ++s){
                        for (int d = 0; d < layer_structure[dst_layer]; ++d){
                            synapse nerve(_random::random() < density_coefficient);
                            nerve.start_loc = std::make_tuple(src_layer, s);
                            nerve.end_loc = std::make_tuple(dst_layer, d);
                            from_layer.push_back(nerve);
                        }
                    }
                }this->nerves.push_back(from_layer);
            }

        }        

        std::vector<double> forward(std::vector<double> input){

            // let the first layer hold the input
            for (int r = 0; r < input.size(); ++r){
                this->brain[0][r].holder = input[r];
            }

            for (int layer_number = 0; layer_number < this->nerves.size(); ++layer_number){
                for (synapse &nerve: this->nerves[layer_number]){
                    neuron *src = &(this->brain[std::get<0>(nerve.start_loc)][std::get<1>(nerve.start_loc)]);
                    neuron *dst = &(this->brain[std::get<0>(nerve.end_loc)][std::get<1>(nerve.end_loc)]);
					if (!(nerve.enabled)){
						// dst->from_origin = false;
						nerve.enabled = false;
						dst->holder = 0;
						continue;
					}dst->holder += src->holder * nerve.weight; 
					// dst->from_origin = true;
                }for (neuron &cell: this->brain[layer_number+1]){
                    cell.forward();
                }
                
            }

            std::vector<double> output;
            int last = this->brain.size()-1;
            for (neuron cell: this->brain[last]){
                output.push_back(cell.holder);
            }

            // reset the holders
            for (int l = 1; l <= last; ++l){
                for (neuron &cell: this->brain[l]){
                    cell.holder = 0;
					cell.from_origin = true;
                }
            }

            return output;
        }

		void mutate(double change, double toggle_probability){

			for (int l = 1; l < this->brain.size(); ++l){
				for (neuron &cell: this->brain[l]){
					cell.mutate(change);
				}
			}

			for (int l = 0; l < this->nerves.size(); ++l){
				for (synapse &nerve: this->nerves[l]){
					nerve.mutate(change);
					if (_random::random() < toggle_probability){
						nerve.enabled = !nerve.enabled;
					}
				}
			}
		}

// 		void drawNetwork(sf::RenderWindow &window){
// 
// 			int x, y, x0, y0;
// 			int side = 15;
// 
// 			for (std::vector<neuron> layer: this->brain){
// 				for (neuron n: layer){
// 					std::tuple<int, int> coor = this->transform(n.location);
// 					x = std::get<0>(coor) + side/2;
// 					y = std::get<1>(coor) + side/2;
// 					x0 = std::get<0>(coor) - side/2;
// 					y0 = std::get<1>(coor) - side/2;
// 					this->drawBox(x, y, x0, y0, window);
// 				}
// 			}
// 			
// 			for (std::vector<synapse> nerve: this->nerves){
// 				for (synapse s: nerve){
// 					if (!s.enabled) continue;
// 					std::tuple<int, int> start_coor = this->transform(s.start_loc);
// 					std::tuple<int, int> end_coor = this->transform(s.end_loc);
// 					this->drawLine(std::get<0>(start_coor), std::get<1>(start_coor), 
// 							std::get<0>(end_coor), std::get<1>(end_coor), window);
// 				}
// 			}
// 		}
// 
// 		void display(int width, int height, std::string window_name){
// 
// 			sf::RenderWindow window(sf::VideoMode(width, height), window_name);
// 
// 			while (window.isOpen()){
// 				sf::Event event;
// 				while (window.pollEvent(event)){
// 					if (event.type == sf::Event::Closed){
// 						window.close();
// 					}
// 				}
// 				window.clear(sf::Color::Black);
// 				this->drawNetwork(window);
// 				window.display();
// 			}
// 		}

		network clone(){
			std::vector<std::vector<neuron>> temp_brain;
			std::vector<std::vector<synapse>> temp_nerves;

			// clone the brain
			for (std::vector<neuron> neuron_layer: this->brain){
				std::vector<neuron> layer;
				for (neuron n: neuron_layer){
					layer.push_back(n.clone());
				}temp_brain.push_back(layer);
			}

			//clone the connections
			for (std::vector<synapse> synapse_layer: this->nerves){
				std::vector<synapse> layer;
				for (synapse s: synapse_layer){
					layer.push_back(s.clone());
				}temp_nerves.push_back(layer);
			}

			network baby;
			baby.brain = temp_brain;
			baby.nerves = temp_nerves;

			return baby;
		}

		int xoffset = 30;
		int yoffset = 30;

		int xspacing = 300;
		int yspacing = 30;

		std::tuple<int, int> transform(std::tuple<int, int> loc){
			int x = this->xoffset + std::get<0>(loc) * this->xspacing;
			int y = this->yoffset + std::get<1>(loc) * this->yspacing;
			return std::make_tuple(x, y);
		}

// 		void drawLine(int x0, int y0, int x, int y, sf::RenderWindow &window){
// 			sf::VertexArray line(sf::Lines, 2);
// 			
// 			line[0].position = sf::Vector2f(x0, y0);
// 			line[0].color = sf::Color::Blue;
// 			line[1].position = sf::Vector2f(x, y);
// 			line[1].color = sf::Color::Blue;
// 			window.draw(line);
// 		}
// 
// 		void drawBox(int x0, int y0, int x1, int y1, sf::RenderWindow &window){
// 			sf::VertexArray box(sf::Quads, 4);
// 
// 			box[0].position = sf::Vector2f(x0, y0);
// 			box[0].color = sf::Color::Red;
// 			
// 			box[1].position = sf::Vector2f(x0, y1);
// 			box[1].color = sf::Color::Red;
// 			
// 			box[2].position = sf::Vector2f(x1, y1);
// 			box[2].color = sf::Color::Red;
// 			
// 			box[3].position = sf::Vector2f(x1, y0);
// 			box[3].color = sf::Color::Red;
// 			window.draw(box);
// 		}

    };
}

namespace genetic{

	feedforward::network crossover(feedforward::network &agent1, 
			feedforward::network &agent2, 
			double c_prob, 
			double m_std,
			double m_prob){

		feedforward::network baby = agent1.clone();

		/*
			Brain of the baby remains the same
			Only the synapses will be changed
		 
		 1) intersection of the two is considered with a probability of 1
		 2) union - untersection [(n1 v n2) - (n1 ^ n2)] of the two is considered with a probability of c_prob
		 3) mutation with a stantard deviation of m_std with a probability of m_prob

			To make it computationally faster...
			1) and 2) can be combined
			n.enabled = (n1.enabled && n2.enabled) || ((n1.enabled ^ n2.enabled) && (random() < c_prob))
		*/

		// main crossover (1 and 2)
		for (int layer_number = 0; layer_number < agent1.nerves.size(); ++layer_number){
			for (int s_number = 0; s_number < agent1.nerves[layer_number].size(); ++s_number){
				bool one = agent1.nerves[layer_number][s_number].enabled;
				bool two = agent2.nerves[layer_number][s_number].enabled;
				baby.nerves[layer_number][s_number].enabled = (one && two) || ((one ^ two) && (_random::random() < c_prob));
			}
		}

		baby.mutate(m_std, m_prob);
		return baby;
	}
}

PYBIND11_MODULE(neat, handle){
	handle.doc() = "test";
	
	// random namespace
	pybind11::module _random = handle.def_submodule("_random", "consists the random functions");
	_random.def("random", &_random::random);
	_random.def("random_checker", &_random::random_checker);
	_random.def("normal0_1", &_random::normal0_1);
	_random.def("normal_checker", &_random::normal_checker);
	_random.def("normal", &_random::normal);

	// feedforward
	pybind11::module feedforward = handle.def_submodule("feedforward", "feedforward neural network, neurons and synapses");
	
	pybind11::class_<feedforward::synapse>(feedforward, "synapse")
		.def(pybind11::init<>())
		.def(pybind11::init<bool>())
		.def("mutate", &feedforward::synapse::mutate)
		.def("clone", &feedforward::synapse::clone)
		.def_readwrite("start_loc", &feedforward::synapse::start_loc)
		.def_readwrite("end_loc", &feedforward::synapse::end_loc)
		.def_readwrite("weight", &feedforward::synapse::weight)
		.def_readwrite("enabled", &feedforward::synapse::enabled)
		;


	pybind11::class_<feedforward::neuron>(feedforward, "neuron")
		.def(pybind11::init<>())
		.def(pybind11::init<double>())
		.def("activation", &feedforward::neuron::activation)
		.def("forward", &feedforward::neuron::forward)
		.def("mutate", &feedforward::neuron::mutate)
		.def("clone", &feedforward::neuron::clone)
		.def_readwrite("bias", &feedforward::neuron::bias)
		.def_readwrite("holder", &feedforward::neuron::holder)
		.def_readwrite("from_origin", &feedforward::neuron::from_origin)
		.def_readwrite("location", &feedforward::neuron::location)
		;

	pybind11::class_<feedforward::network>(feedforward, "network")
		.def(pybind11::init<>())
		.def(pybind11::init<std::vector<int>, double>())
		.def("forward", &feedforward::network::forward)
		.def("mutate", &feedforward::network::mutate)
		//.def("drawNetwork", &feedforward::network::drawNetwork)
		//.def("display", &feedforward::network::display)
		.def("clone", &feedforward::network::clone)
		.def("transform", &feedforward::network::transform)
		//.def("drawLine", &feedforward::network::drawLine)
		//.def("drawBox", &feedforward::network::drawBox)
		.def_readwrite("brain", &feedforward::network::brain)
		.def_readwrite("nerves", &feedforward::network::nerves)
		.def_readwrite("xoffset", &feedforward::network::xoffset)
		.def_readwrite("yoffset", &feedforward::network::yoffset)
		.def_readwrite("xspacing", &feedforward::network::xspacing)
		.def_readwrite("yspacing", &feedforward::network::yspacing)
		;

	// genetic namespace
	pybind11::module genetic = handle.def_submodule("genetic", "contains the code for genetic crossover");
	genetic.def("crossover", &genetic::crossover);

}

