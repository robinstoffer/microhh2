#include <iostream>
#include <memory>

class Animal
{
    public:
        virtual ~Animal() {};
        virtual void make_sound() = 0;
        virtual void set_age(const int age_in) { age = age_in; }
        virtual void print_age() { std::cout << "Age = " << age << std::endl; }

    private:
        int age;
};

class Dog : public Animal
{
    public:
        Dog() {}
        ~Dog() { std::cout << "Deleting Dog" << std::endl; }
        void make_sound() { std::cout << "Woof" << std::endl; }
        void set_age(const int age_in) { Animal::set_age(age_in); }
        void wag_tail() { std::cout << "Wagging tail" << std::endl; }
};

class Cat : public Animal
{
    public:
        Cat() {}
        ~Cat() { std::cout << "Deleting Cat" << std::endl; }
        void make_sound() { std::cout << "Meaw" << std::endl; }
        void set_age(const int age_in) { Animal::set_age(age_in); }
        void wag_tail() { std::cout << "I refuse, I am not a dog." << std::endl; }
};

void make_sound_interface(Animal& animal)
{
    animal.make_sound();
}

/*
std::unique_ptr<Animal> animal_factory(const int a)
{
    std::unique_ptr<Animal> animal;
    if (a == 0)
        animal = std::make_unique<Dog>();
    else if (a == 1)
        animal = std::make_unique<Cat>();
    else
        std::cout << "Illegal animal" << std::endl;
}
*/

int main()
{
    Dog dog;
    dog.make_sound();
    dog.set_age(3);
    dog.print_age();

    int a;
    std::cin >> a;

    // std::unique_ptr<Animal> animal;
    Animal* animal;
    if (a == 0)
        // animal = std::make_unique<Dog>();
        animal = new Dog();
    else if (a == 1)
        // animal = std::make_unique<Cat>();
        animal = new Cat();
    else
        std::cout << "Illegal animal" << std::endl;

    // These two are the same.
    animal->make_sound();
    (*animal).make_sound();

    make_sound_interface(*animal);

    Dog* dog_ptr = dynamic_cast<Dog*>(animal);
    if (dog_ptr == nullptr)
        std::cout << "Oops...." << std::endl;
    else
        dog_ptr->wag_tail();

    return 0;
}
